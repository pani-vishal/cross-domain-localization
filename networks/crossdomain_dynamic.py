import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import shutil

class DynamicNet(torch.nn.Module):
    def __init__(self, data, list_domains=["night"], use_cuda=True):
        """
        Model to handle multi-domains via separate extractors and a common part.
        Training handled in a rotating fashion.
        """
        super(DynamicNet, self).__init__()
        self.data = data
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.list_domains = list_domains
        # All the extractors are the first half of a ResNet34 model, stored as a dictionary
        self.dynamic_extractors = {}
        for domain in list_domains:
            self.dynamic_extractors[domain] = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:6]).to(self.device)

        print("Keys of dynamic extractors:", self.dynamic_extractors.keys())
        # By default extractor is set as the first element of list_domains
        self.set_feature_extractor(list_domains[0])

        # self.half_second is the common part of the model
        self.half_second = nn.Sequential(*list(models.resnet34(pretrained=True).children())[6:-1]).to(self.device)
        self.fc1 = nn.Linear(1024, 512).to(self.device)
        self.bn1 = nn.BatchNorm1d(num_features=512).to(self.device)
        self.fc2 = nn.Linear(512, 2).to(self.device)
        self.relu = nn.ReLU(inplace=False).to(self.device)

        self.epoch = 0
        self.epochs = None
        self.max_lr = None

        self.opt_static = None
        self.sched_static = None
        self.dict_opt_dynamic = {}
        self.dict_sched_dynamic = {}


    def forward(self, static_feats, dynamic_img):
        # Note: static embeddings are input right at the end of the model
        dynamic_feats = self.half_dynamic(dynamic_img)
        dynamic_feats = self.half_second(dynamic_feats)
        dynamic_feats = torch.flatten(dynamic_feats, 1)
        out = torch.cat((static_feats, dynamic_feats), axis=1)
        out = self.fc1(out)
        out = self.relu(self.bn1(out))
        out = self.fc2(out)
        out = nn.functional.log_softmax(out, dim=1)
        return out


    def _init_optimizers_schedulers(self, max_lr, epochs, div_factor=1.5):
        """
        Function for creating dictionary of optimizers for different branches and common part.
        Optimizer have differential learning rate determined by the div_factor.
        The OneCycleLR schedulers are also defined at the end.
        """

        len_dynamic_layers = len(self.half_dynamic)
        len_half_second_layers = len(self.half_second)

        # Optimizer for the common part of the model
        self.opt_static = optim.AdamW([
                            {'params': self.half_second[:len_half_second_layers//2].parameters(), 'lr': max_lr / div_factor**2},
                            {'params': self.half_second[len_half_second_layers//2:].parameters(), 'lr': max_lr / div_factor},
                            {'params': self.fc1.parameters(), 'lr': max_lr / div_factor},
                            {'params': self.bn1.parameters(), 'lr': max_lr},
                            {'params': self.fc2.parameters(), 'lr': max_lr},
                       ], lr=max_lr)
        
        list_lrs = [
                    max_lr / div_factor**2,
                    max_lr / div_factor,
                    max_lr / div_factor,
                    max_lr,
                    max_lr
                  ]
        # Scheduler for the common part of the model
        self.sched_static = lr_scheduler.OneCycleLR(self.opt_static, max_lr=list_lrs, epochs=epochs, steps_per_epoch=len(self.data["train"]), div_factor=9)

        # Creating dictionary of optimizers and schedulers for the different branches
        for domain in self.list_domains:
            self.dict_opt_dynamic[domain] = optim.AdamW([{'params': self.dynamic_extractors[domain][:len_dynamic_layers//2].parameters(), 'lr': max_lr / div_factor**4},
                                                          {'params': self.dynamic_extractors[domain][len_dynamic_layers//2:].parameters(), 'lr': max_lr / div_factor**3}], 
                                                         lr=div_factor**3)
            list_lrs = [
                    max_lr / div_factor**4,
                    max_lr / div_factor**3
                ]
            self.dict_sched_dynamic[domain] = lr_scheduler.OneCycleLR(self.dict_opt_dynamic[domain], 
                                                                       max_lr=list_lrs, 
                                                                       epochs=epochs, 
                                                                       steps_per_epoch=len(self.data["train"]), 
                                                                       div_factor=9)


    def _optim_zero_grad(self):
        self.opt_static.zero_grad()
        for domain in self.list_domains:
            self.dict_opt_dynamic[domain].zero_grad()


    def _static_grad_div(self, div_factor):
        """ For dividing the params of common layers by number of domains """
        for param_group in self.opt_static.param_groups:
            for params in param_group['params']:
                params.grad.div_(div_factor)

    

    def _set_eval(self):
        """
        Since forward function's layer changes, 
        we need to set all the layers to eval manually
        """
        for domain in self.dynamic_extractors:
            self.dynamic_extractors[domain].eval()
        self.half_second.eval()
        self.fc1.eval()
        self.bn1.eval()
        self.fc2.eval()
        self.relu.eval()

    def _set_train(self):
        """
        Since forward function's layer changes, 
        we need to set all the layers to train manually
        """
        for domain in self.dynamic_extractors:
            self.dynamic_extractors[domain].train()
        self.half_second.train()
        self.fc1.train()
        self.bn1.train()
        self.fc2.train()
        self.relu.train()


    def _train(self, epoch):
        total_loss = 0
        self._set_train()
        for _, dict_sample in tqdm(enumerate(self.data['train']), total=len(self.data['train'])):
            self._optim_zero_grad()
            batch_loss = 0
            # Rotating training: for each domain we set the self.half_dynamic as the extractor of that domain.
            # This replaces the previous extractor in the models's graph.
            # So on loss.backward() only the common part and that specific branch's gradients get updated
            # self.dict_opt_dynamic[domain].step() updates only that specific branch
            # Also, the gradients of the common part keeps on accumalating
            for domain in dict_sample.keys():
                self.set_feature_extractor(domain)
                (img1, img2), target = dict_sample[domain]
                target = target.to(self.device)
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)   
                output = self(img1, img2)
                loss = nn.functional.nll_loss(output, target)
                batch_loss += loss.item()
                loss.backward()
                self.dict_opt_dynamic[domain].step()
            
            # Common part's gradient is divided by number of domains, therefore equating to the mean of accumulated gradients.
            self._static_grad_div(len(dict_sample.keys()))
            self.opt_static.step()

            batch_loss /= len(dict_sample.keys())
            total_loss += batch_loss

        mean_loss = total_loss / len(self.data['train'])
        print(f'Train Epoch: {epoch}, Mean loss: {mean_loss}')


    def _validate(self):
        test_loss = 0
        correct = 0
        self._set_eval()
        with torch.no_grad():
            for batch_idx, dict_sample in tqdm(enumerate(self.data['validation']), total=len(self.data['validation'])):
                batch_loss = 0
                for domain in dict_sample.keys():
                    self.set_feature_extractor(domain)
                    (img1, img2), target = dict_sample[domain]
                    target = target.to(self.device)
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    output = self(img1, img2)
                    batch_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                
                batch_loss /= len(dict_sample.keys())
                test_loss += batch_loss

        test_loss /= len(self.data['validation'].dataset)
        correct /= len(dict_sample.keys())

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data['validation'].dataset),
            100. * correct / len(self.data['validation'].dataset)))


    def _fit_from_start_to_end(self, epochs, start, end):
        """Model's fit procedure from a given start to end epoch"""
        for epoch in range(start, end):
            self.epoch = epoch
            self._train(epoch)
            self._validate()
            self.sched_static.step()
            for scheduler in self.dict_sched_dynamic.values():
                scheduler.step()

    def fit(self, max_lr=1e-4, epochs=5, continue_train=True):
        """Wrapper for model's fit procedure"""
        self.epochs = epochs
        self.max_lr = max_lr
        # To train from scratch
        if self.epoch == 0:
            print(f"Starting a new procedure, from epoch 1 to {epochs}...")
            self._init_optimizers_schedulers(max_lr, epochs)
            self._fit_from_start_to_end(epochs, 1, epochs+1)
        # To continue training from checkpoint uptil given end epoch
        elif continue_train:
            print(f"Resuming procedure, from epoch {self.epoch} to {epochs}...")
            self._fit_from_start_to_end(epochs, self.epoch, epochs+1)
        # To continue training from checkpoint
        else:
            print(f"Resuming procedure, from epoch {self.epoch} to {self.epoch + epochs + 1}...")
            self._fit_from_start_to_end(epochs, self.epoch, self.epoch+epochs+1)


    
    def set_feature_extractor(self, domain="night"):
        self.half_dynamic = self.dynamic_extractors[domain]


    def set_dynamic_extractors(self, dynamic_extractors):
        self.dynamic_extractors = dynamic_extractors


    def save(self, folder_path):
        """Saving the model in a modular fashion"""
        # Handling path
        if os.path.isdir(folder_path):
            print(f"{folder_path} already exists. Delete it?(y/n)")
            decision = input()
            if decision in ['y', 'Y', 'Yes', 'yes']:
                shutil.rmtree(folder_path)
                os.mkdir(folder_path)
            else:
                print("Cool. Not saving anything.")
                return
        else:
            os.mkdir(folder_path)

        # Save dynamic extractors
        for domain in self.dynamic_extractors:
            model_name = f'dynamic_{domain}'
            model_path = os.path.join(folder_path, model_name)
            extractor = self.dynamic_extractors[domain]
            opt = self.dict_opt_dynamic[domain]
            sched = self.dict_sched_dynamic[domain]

            torch.save({
            'model_state_dict': extractor.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            }, model_path)

        model_name = 'static_half'
        model_path = os.path.join(folder_path, model_name)
        opt = self.opt_static
        sched = self.sched_static

        torch.save({
            'epoch': self.epoch,
            'max_lr': self.max_lr,
            'epochs': self.epochs,
            'half_second_state_dict': self.half_second.state_dict(),
            'fc1_state_dict': self.fc1.state_dict(),
            'bn1_state_dict': self.bn1.state_dict(),
            'fc2_state_dict': self.fc2.state_dict(),
            'relu_state_dict':self.relu.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
        }, model_path)

        print(f"Saved at {folder_path} successfully!")
        

    def load(self, folder_path, flag_inference=True):
        """Loading the model from a given path"""
        if not os.path.isdir(folder_path):
            print("No folder found.")
            return
        elif flag_inference:
            # Load dynamic parts
            for domain in self.dynamic_extractors:
                model_name = f'dynamic_{domain}'
                model_path = os.path.join(folder_path, model_name)
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                self.dynamic_extractors[domain].load_state_dict(checkpoint['model_state_dict'])
            # Load each static part
            model_name = 'static_half'
            model_path = os.path.join(folder_path, model_name)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            self.half_second.load_state_dict(checkpoint['half_second_state_dict'])
            self.fc1.load_state_dict(checkpoint['fc1_state_dict'])
            self.bn1.load_state_dict(checkpoint['bn1_state_dict'])
            self.fc2.load_state_dict(checkpoint['fc2_state_dict'])
            self.relu.load_state_dict(checkpoint['relu_state_dict'])
            self.eval()
        else:
            # Load each static part
            model_name = 'static_half'
            model_path = os.path.join(folder_path, model_name)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.epoch = checkpoint['epoch']
            self.epochs = checkpoint['epochs']
            self.max_lr = checkpoint['max_lr']
            self._init_optimizers_schedulers(self.max_lr, self.epochs, div_factor=1.5)

            self.half_second.load_state_dict(checkpoint['half_second_state_dict'])
            self.fc1.load_state_dict(checkpoint['fc1_state_dict'])
            self.bn1.load_state_dict(checkpoint['bn1_state_dict'])
            self.fc2.load_state_dict(checkpoint['fc2_state_dict'])
            self.relu.load_state_dict(checkpoint['relu_state_dict'])
            self.opt_static.load_state_dict(checkpoint['optimizer_state_dict'])
            self.sched_static.load_state_dict(checkpoint['scheduler_state_dict'])
            # Load dynamic parts
            for domain in self.dynamic_extractors:
                model_name = f'dynamic_{domain}'
                model_path = os.path.join(folder_path, model_name)
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

                self.dynamic_extractors[domain].load_state_dict(checkpoint['model_state_dict'])
                self.dict_opt_dynamic[domain].load_state_dict(checkpoint['optimizer_state_dict'])
                self.dict_sched_dynamic[domain].load_state_dict(checkpoint['scheduler_state_dict'])
            self.train()

        print(f'Loaded from {folder_path} sucessfully!')