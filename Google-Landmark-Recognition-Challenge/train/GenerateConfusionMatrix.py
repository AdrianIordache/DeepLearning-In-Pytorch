import gc
import cv2
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from IPython.display import display
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, models, transforms

import albumentations as A


def get_value_from_index(l, index):
  return l[index]

def get_index_from_value(l, value):
  for i in range(len(l)):
    if value == l[i]:
      return i

class LandmarkDataset(Dataset):
    def __init__(self, features, labels, transform = None, augmentation = None):
        self.features = features
        self.labels   = labels
        self.transform = transform
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):

        x = Image.open(self.features[index])
        y = self.labels[index]
        
        if self.transform is not None:
            x = self.transform(x)
            
        if self.augmentation is not None:
            x = self.augmentation(image = np.array(x))["image"]
            x = np.transpose(x, (2, 0, 1)).astype(np.float32)
            x = torch.tensor(x, dtype=torch.float)

        # print(x.shape)
        return x, y


PATH_TO_PRETRAINED_MODEL = "/home/adrian/Desktop/Python/Personal/Deep/Final-Project/models/30/resnet18_pretrained_no_augmentation_0.980502724647522.pt"

most_frequent_10 = [2061, 2743, 5376, 5554, 6051, 6599, 6651, 6696, 9633, 9779]
most_frequent_20 = [1553, 2061, 2743, 2949, 4352, 4987, 5376, 5554, 6051, 6599, 6651, 6696, 8063, 8429, 9633, 9779, 10900, 11784, 12220, 13526]
most_frequent_30 = [428, 1553, 2061, 2338, 2743, 2949, 3804, 3924, 4352, 4987, 5376, 5554, 6051, 6599, 6651, 6696, 7092, 8063, 8429, 9029, 9633, 9779, 10045, 10184, 10900, 11784, 12172, 12220, 12718, 13526]
most_frequent_50 = [428, 1553, 1847, 1878, 2044, 2061, 2338, 2449, 2743, 2870, 2949, 3283, 3497, 3804, 3924, 4352, 4987, 5367, 5376, 5554, 5955, 6051, 6231, 6599, 6651, 6696, 7092, 7172, 7661, 8063, 8169, 8429, 9029, 9434, 9633, 9779, 10026, 10033, 10045, 10184, 10900, 10932, 11249, 11784, 12172, 12220, 12718, 13170, 13526, 13653]

path_to_images = "train_frequent_30/*"

data_dict = {
  "Paths": [],
  "Labels": []
}

for subdir in sorted(glob.glob(path_to_images)):
	label = int(subdir.split('/')[-1])
	for image in sorted(glob.glob(subdir + "/*")):
		data_dict["Paths"].append(image)
		data_dict["Labels"].append(get_index_from_value(most_frequent_30, label))

data = pd.DataFrame.from_dict(data_dict)
display(data.head(n = 10))

print("On GPU: " + str(torch.cuda.is_available()))

print("Number of samples:", data.shape[0])
print("Labels:", most_frequent_30)
print("Encoded Labels:", set(data["Labels"].values))


pickle_idx = open("dataset_indices_30.pickle", "rb")
dataset_indices = pickle.load(pickle_idx)

X_train, y_train = data.iloc[dataset_indices["train_idx"]]["Paths"].values, data.iloc[dataset_indices["train_idx"]]["Labels"].values
X_valid, y_valid = data.iloc[dataset_indices["valid_idx"]]["Paths"].values, data.iloc[dataset_indices["valid_idx"]]["Labels"].values
X_holdout, y_holdout = data.iloc[dataset_indices["holdout_idx"]]["Paths"].values, data.iloc[dataset_indices["holdout_idx"]]["Labels"].values


train_transforms = transforms.Compose([
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])

valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained = False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 30)

model.load_state_dict(torch.load(PATH_TO_PRETRAINED_MODEL))

model.to(device)

dataset = LandmarkDataset(X_holdout, y_holdout, transform = valid_transforms)
loader = DataLoader(dataset, batch_size = 48, shuffle = True, drop_last = True)

y_true = []
y_predict = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(loader):

        if batch_idx % 1000 == 0:
            print("Holdout Batch: " + str(batch_idx) + " from: " + str(len(loader)))

        model.eval()

        images = images.to(device)
        labels = labels.to(device)

        output = model(images.float())
        top_class = torch.argmax(output, dim = 1)

        y_true.extend(labels.detach().cpu().numpy())
        y_predict.extend(top_class.detach().cpu().numpy())
        

#print(y_true)
#print(y_predict)
print(len(y_true))
print(len(y_predict))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_true, y_predict)
plt.figure(figsize=(20, 20))
sns.heatmap(confusion_matrix, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix_50.png')
plt.show();


print("Scores for each class")
print(confusion_matrix.diagonal()/confusion_matrix.sum(axis=1))