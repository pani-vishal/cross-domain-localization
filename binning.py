"""
Script contains all the functions relevant to inference via binning
"""
import numpy as np
import torch


def get_similarity(model, embed, dynamic):
    """To get similarity score between static embedding and the dynamic image"""
    return np.exp(model(embed, dynamic).cpu().detach().numpy()[0][1])


def get_sequence_similarity(model, night_images_sequence, day_images_sequence):
    """Get sum of similarity score of sequences of images"""
    return sum(get_similarity(model, day_image, night_image) \
                for night_image, day_image in zip(night_images_sequence, day_images_sequence))


def get_input_embedding(embedding):
    embedding = torch.from_numpy(np.asarray(embedding)).float()
    return torch.unsqueeze(embedding, dim=0)


def get_input_img(img, normalize, to_tensor):
    img = normalize(to_tensor(img))
    return torch.unsqueeze(img, dim=0)

def get_bin_number_from_image_number(image_number, bin_size=50):
  return image_number // bin_size


def get_bin_ids(num_imgs, bin_size, list_corner=None):
    """Return bin representative images as a list"""
    return list(range(0, num_imgs, bin_size))