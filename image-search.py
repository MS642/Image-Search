import sys
import argparse
import torch
import glob
import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.models import vgg16
from torchvision import transforms

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
            name = name.strip()
            images_db[name].add(image)
            for word in name.split(' '):
                images_db[word.lower()].add(image)

def predict_images(dir, verbose):
    # Predict labels with associated probabilities for unseen images
    images = glob.glob(dir)
    for image in images:
        img = Image.open(image)
        df = classify_image(img)
        if verbose:
            print(image)
            print(df)
        add_to_cache(image, df)


def _exit_program(event):
    sys.exit()
def main():

    parser = argparse.ArgumentParser(
     prog='Image search',
     description='''The program uses ML library to define the probability what 
     each image could be in the given directory. Then creating a map of keywords
     to each image of the program at startup.
     The program then takes user keyword and outputs images with those keywords ''',
     epilog='''
         The challenge was open ended and I figured with live DB, 
         having auth secrets in github wouldn't be a good idea; 
         and therefore went with a simple python solution. 
         Currently I am studing ML and AI so was interested in 
         trying to apply what I am learning for the challenge.''')
    parser.add_argument('--keyword', '-k', action="store", type=str, nargs='?', required=True,
                        help='search keyword')
    parser.add_argument('--directory', '-d', action="store", default="data/test_images/*.*",
                        help='valid directory of where the images are located')
    parser.add_argument('--verbose', '-v', action="store_true",
                        help='to allow verbose output')

    args = parser.parse_args()
    print(args)
    predict_images(args.directory, args.verbose)
    search_keyword = args.keyword.strip().lower()
    if search_keyword in images_db:
        result_size = len(images_db[search_keyword])
        fig = plt.figure(figsize=(result_size, result_size))
        for ind, image in enumerate(images_db[search_keyword]):
            img = Image.open(image)
            img.load()
            fig.add_subplot(result_size//2 +1, result_size//2 +1, ind+1)
            plt.imshow(img)
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        axcut = plt.axes([0.4, 0.0, 0.2, 0.075])
        bcut = plt.Button(axcut, 'Exit Search Results', color='grey', hovercolor='red')
        bcut.on_clicked(_exit_program)
        plt.show()
    else:
        print("0 results found for " + search_keyword + " try "
                "searching something else.")
        
        print("Keywords for image results: " )
        print(list(images_db.keys()))

if __name__ == "__main__":
    main()