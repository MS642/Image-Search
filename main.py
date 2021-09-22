from typing import Collection
import pandas as pd
import numpy as np
# import re 
from PIL import Image
from torchvision.models import vgg16
from torchvision import transforms
import torch

# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

import glob
import collections
# visualization
# import graphviz
import matplotlib.pyplot as plt
# from imageio import imread
plt.rcParams["font.size"] = 16
images_db = collections.defaultdict(set)
def classify_image(img, topn = 4):

    clf = vgg16(pretrained=True)
    preprocess = transforms.Compose([
                 transforms.Resize(299),
                 transforms.CenterCrop(299),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),])

    with open('data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    clf.eval()
    output = clf(batch_t)
    _, indices = torch.sort(output, descending=True)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    d = {'Class': [classes[idx] for idx in indices[0][:topn]], 
         'Probability score': [np.round(probabilities[0, idx].item(),3) for idx in indices[0][:topn]]}
    df = pd.DataFrame(d, columns = ['Class','Probability score'])
    return df

def add_to_cache(image, df):
    string_probability = set()
    for i in range(len(df)):
        class_names = df['Class'][i].split(',')
        # probability = df['Probability Score'][i]  # TODO: improve output sorting order
        for name in class_names:
            images_db[name].add(image)
            for word in name.split(' '):
                images_db[word].add(image)


# Predict labels with associated probabilities for unseen images
images = glob.glob("data/test_images/*.*")
for image in images:
    img = Image.open(image)
    # img.load()
    # plt.imshow(img)
    # plt.show()
    df = classify_image(img)
    # print(df.to_string(index=False))
    add_to_cache(image, df)
    # print(df)
    # print("--------------------------------------------------------------")

exit = False
while not exit:
    val = input("Enter image search: ")
    if val == 'exit':
        exit = True
    else:
        w = 10
        h = 10
        result_size = len(images_db[val])
        # print(images_db)
        # print(val)
        # print(result_size)
        fig = plt.figure(figsize=(result_size, result_size))
        for ind, image in enumerate(images_db[val]):
            img = Image.open(image)
            img.load()
            fig.add_subplot(result_size//2 +1, result_size//2 +1, ind+1)
            plt.imshow(img)
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
        plt.show()
