import os
import h5py
import random
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from plotly.offline import iplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_image_files(path):
    """Get list of images from given path"""
    list_images = [os.path.join(path, x) for x in os.listdir(path)]
    return list_images


def get_image_arr(path):
    """Get image numpy array from .h5 file"""
    with h5py.File(path, 'r') as hf:
        image_arr = hf.get('images')[()]
    return image_arr


def get_input_tensor_img(img, normalize, to_tensor):
    """Convert img arr to input tensor image"""
    img = normalize(to_tensor(img))
    return torch.unsqueeze(img, dim=0)


def generate_static_embeddings(model, arr_images, embedding_size=256):
    """Generate static embeddings from embedding network"""
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                     std=(0.229, 0.224, 0.225))

    embeddings = np.empty((arr_images.shape[0], embedding_size))
    with torch.no_grad():
        model.eval()
        for idx in tqdm(range(arr_images.shape[0])):
          img = get_input_tensor_img(arr_images[idx], normalize, to_tensor)
          embeddings[idx] = model.get_embedding(img).data.cpu().numpy()
    return embeddings


def get_different_image_no(img_number, dataset_size, dist=30):
    """Get a image idx away from provided img_number as idx of an array""" 
    past = (img_number - dist) % dataset_size
    future = (img_number + dist) % dataset_size

    samples = None
    if past < future: samples = list(range(past)) + list(range(future + 1, dataset_size))
    else: samples = list(range(future + 1, past))

    return random.sample(samples, k=1)[0] % dataset_size


def get_different_close_image_no(img_number, dataset_size, dist=30, threshold=10):
    """Get a image idx away from provided img_number but within a threshold""" 
    past_low = (img_number - (dist+dataset_size//threshold)) % dataset_size
    past = (img_number - dist) % dataset_size
    future = (img_number + dist) % dataset_size
    future_high = (img_number + (dist+dataset_size//threshold)) % dataset_size
    past_samples = None
    fut_samples = None
    if past_low < past and past_low < img_number: past_samples = list(range(past_low, past))
    if future < future_high and img_number < future_high: fut_samples = list(range(future+1, future_high+1))

    if not fut_samples:
        return random.sample(list(range(past_low, past)), k=1)[0] % dataset_size
    if  not past_samples:
        return random.sample(list(range(future+1, future_high+1)), k=1)[0] % dataset_size
    return random.sample(past_samples + fut_samples, k=1)[0] % dataset_size


def get_similar_image_no(img_number, dataset_size, dist=20, loop=False):
    """Get a image idx close to provided img_number as idx of an array""" 
    if loop:
        past = (img_number - dist) % dataset_size
        future = (img_number + dist) % dataset_size
    else:
        if img_number < dist:
            past = 0
            future = dist
        elif dataset_size - img_number < dist:
            past = - (dist + 1)
            future = -1
        else:
            past = (img_number - dist)
            future = (img_number + dist)

    samples = None
    if past < future: samples = list(range(past, future + 1))
    else: samples = list(range(past, dataset_size)) + list(range(future + 1))
    return random.sample(samples, k=1)[0] % dataset_size


def get_diff_image_idx(list_imgs, len_ds):
    """Get list of different idx pairs"""
    return [get_different_image_no(idx, len_ds)
            for idx in range(len_ds)]


def get_diff_close_image_idx(list_imgs, len_ds):
    """Get list of different but close idx pairs"""
    return [get_different_close_image_no(idx, len_ds)
            for idx in range(len_ds)]


def get_same_image_idx(list_imgs, len_ds):
    """Get list of similar idx pairs"""
    return [get_similar_image_no(idx, len_ds)
            for idx in range(len_ds)]


def plot_triplet_ex(ds, rows, cols=2,shuffle=True):
    """
    Code adapted from: https://github.com/adambielski/siamese-triplet
    Display first few examples from triplet dataset
    """
    mean, std = torch.tensor(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    mean, std = mean[...,None,None], std[...,None,None]

    def deprocess(imgt):
        """Convert from torch tensor to numpy for display"""
        imgt = torch.clamp((imgt*std+msean)*255., min=0).numpy().astype(np.uint8)
        return np.transpose(imgt, (1,2,0))

    nshow = rows*cols
    _, axs = plt.subplots(rows, cols, figsize=(16,8))
    for _ in range(nshow):
        i = _ if not shuffle else np.random.randint(len(ds))
        img = np.concatenate((deprocess(ds[i][0][0]), deprocess(ds[i][0][1]), deprocess(ds[i][0][2])), axis=1)
        axs.flatten()[_].set_title('anchor | postive | negative')
        axs.flatten()[_].axis('off')
        axs.flatten()[_].imshow(img)


def plot_crossdomain_ex(ds, stats_static, stats_domain, rows, cols=2,shuffle=True, domain='night'):
    """
    Code adapted from: https://github.com/adambielski/siamese-triplet
    Display first few examples from dataset
    """

    def deprocess(imgt, stats):
        """Convert from torch tensor to numpy for display"""
        imgt = torch.clamp((imgt*stats[1]+stats[0])*255., min=0).numpy().astype(np.uint8)
        return np.transpose(imgt, (1,2,0))

    nshow = rows*cols
    fig, axs = plt.subplots(rows, cols, figsize=(16,8))
    for _ in range(nshow):
        i = _ if not shuffle else np.random.randint(len(ds))
        img = np.concatenate((deprocess(ds.get_img_pairs(i)[domain][0][0], stats_static), 
                              deprocess(ds.get_img_pairs(i)[domain][0][1], stats_domain)), 
                             axis=1)
        axs.flatten()[_].set_title(ds[i][domain][1]);axs.flatten()[_].axis('off')
        axs.flatten()[_].imshow(img)


def plot_pca_embeddings_2d(df, xlim=None, ylim=None):
    """2D Plot of PCA embeddings"""
    plt.figure(figsize=(10,10)) 
    fig = px.scatter(df, x='x', y='y', color='labels', size_max=5)
    iplot(fig)

def plot_pca_embeddings_3d(df, xlim=None, ylim=None):
    """3D Plot of PCA embeddings"""
    plt.figure(figsize=(10,10)) 
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='labels', size_max=5)
    iplot(fig)


def plot_pca(embeddings, dim=3):
    """Plot the PCA of the input embeddings"""
    x = StandardScaler().fit_transform(embeddings)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, 
                            columns = ['x', 'y', 'z'])
    principalDf['labels'] = np.arange(0, embeddings.shape[0])
    if dim == 3:
        plot_pca_embeddings_3d(principalDf)
    else:
        plot_pca_embeddings_2d(principalDf)