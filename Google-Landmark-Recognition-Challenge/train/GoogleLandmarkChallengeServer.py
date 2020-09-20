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

most_frequent_10 = [2061, 2743, 5376, 5554, 6051, 6599, 6651, 6696, 9633, 9779]
most_frequent_20 = [1553, 2061, 2743, 2949, 4352, 4987, 5376, 5554, 6051, 6599, 6651, 6696, 8063, 8429, 9633, 9779, 10900, 11784, 12220, 13526]
most_frequent_30 = [428, 1553, 2061, 2338, 2743, 2949, 3804, 3924, 4352, 4987, 5376, 5554, 6051, 6599, 6651, 6696, 7092, 8063, 8429, 9029, 9633, 9779, 10045, 10184, 10900, 11784, 12172, 12220, 12718, 13526]

path_to_images = "train_frequent_20/*"

data_dict = {
  "Paths": [],
  "Labels": []
}

for subdir in sorted(glob.glob(path_to_images)):
    label = int(subdir.split('/')[-1])
    for image in sorted(glob.glob(subdir + "/*")):
        data_dict["Paths"].append(image)
        data_dict["Labels"].append(get_index_from_value(most_frequent_20, label))

data = pd.DataFrame.from_dict(data_dict)
display(data.head(n = 10))

print("On GPU: " + str(torch.cuda.is_available()))

print("Number of samples:", data.shape[0])
print("Labels:", most_frequent_20)
print("Encoded Labels:", set(data["Labels"].values))


# From Dataset Indices to stay the same

# X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(data["Paths"], data["Labels"], test_size = 0.01, shuffle = True, random_state = 42)

# X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, train_size = 0.8, random_state = 42, shuffle = True)

# print(X_train.index.values,  y_train.index.values)

# dataset_indices = {
#     "train_idx"  : X_train.index.values,
#     "valid_idx"  : X_valid.index.values,
#     "holdout_idx": X_holdout.index.values
# }

# pickle_idx = open("dataset_indices.pickle", "wb")
# pickle.dump(dataset_indices, pickle_idx)
# pickle_idx.close()


pickle_idx = open("dataset_indices_20.pickle", "rb")
dataset_indices = pickle.load(pickle_idx)

X_train, y_train = data.iloc[dataset_indices["train_idx"]]["Paths"].values, data.iloc[dataset_indices["train_idx"]]["Labels"].values
X_valid, y_valid = data.iloc[dataset_indices["valid_idx"]]["Paths"].values, data.iloc[dataset_indices["valid_idx"]]["Labels"].values
X_holdout, y_holdout = data.iloc[dataset_indices["holdout_idx"]]["Paths"].values, data.iloc[dataset_indices["holdout_idx"]]["Labels"].values

train_transforms = None
valid_transforms = None

train_transforms = transforms.Compose([
     transforms.Resize((224, 224)),
     # transforms.RandomHorizontalFlip(p=0.5),
     # transforms.RandomRotation(degrees=(-90, 90)),
     # transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])

valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])




print(train_transforms)
print(valid_transforms)

train_aug = None

# train_aug = A.Compose({
#         A.HorizontalFlip(p=0.5),
#         A.Rotate(limit=(-90, 90)),
#         A.VerticalFlip(p=0.5),
#         A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         })

trainset = LandmarkDataset(X_train, y_train, transform = train_transforms, augmentation = train_aug)
validset = LandmarkDataset(X_valid, y_valid, transform = valid_transforms)

trainloader = DataLoader(trainset, batch_size = 96, shuffle = True, drop_last = True)
validloader = DataLoader(validset, batch_size = 96, shuffle = True, drop_last = True)


# for img, labels in trainloader:
#     inp = img[0]
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     plt.show()
#     break

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)

#num_ftrs = model.fc.in_features

#model.fc = nn.Linear(num_ftrs, 10)

for param in model.parameters():
    param.requires_grad = True

model.classifier[3].out_features = 256
model.classifier[6].in_features  = 256
model.classifier[6].out_features = 20

for param in model.classifier.parameters():
     param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


model.to(device)

best_model = None
best_accuracy = 0

epochs = 10
train_losses, train_accuracies, valid_losses, valid_accuracies = [], [], [], []
for e in range(epochs):
    running_loss = 0
    train_accuracy = 0

    for batch_idx, (images, labels) in enumerate(trainloader):

        if batch_idx % 1000 == 0:
          print("Train Batch: " + str(batch_idx) + " from: " + str(len(trainloader)))

        model.train()
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        output = model(images.float())
        loss = criterion(output, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


        top_class = torch.argmax(output, dim = 1)

        equals = top_class == labels.view(*top_class.shape)

        train_accuracy += torch.mean(equals.type(torch.FloatTensor))

        del images, labels
        torch.cuda.empty_cache()

    else:
        
        valid_loss = 0
        valid_accuracy = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(validloader):
                
                if batch_idx % 1000 == 0:
                  print("Valid Batch: " + str(batch_idx) + " from: " + str(len(validloader)))

                model.eval()
                
                images = images.to(device)
                labels = labels.to(device)

                output = model(images.float())
                valid_loss += criterion(output, labels.long())

                top_class = torch.argmax(output, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                valid_accuracy += torch.mean(equals.type(torch.FloatTensor))

                del images, labels
                torch.cuda.empty_cache()

        train_losses.append(running_loss/len(trainloader))
        train_accuracies.append(train_accuracy.item()/len(trainloader))

        valid_losses.append(valid_loss.item()/len(validloader))
        valid_accuracies.append(valid_accuracy.item()/len(validloader))
        
        if (valid_accuracy / len(validloader)) > best_accuracy:
            best_model = model
            best_accuracy = (valid_accuracy / len(validloader))
        

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Train Accuracy: {:.3f}".format(train_accuracy/len(trainloader)),
              "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
              "Valid Accuracy: {:.3f}".format(valid_accuracy/len(validloader)))



name = "vgg16_pretrained_no_augmentation_"

print("Best Model with Valid Accuracy:", best_accuracy.item())
torch.save(best_model.state_dict(), name + str(best_accuracy.item()) + ".pt")


del model
torch.cuda.empty_cache()


best_model.to(device)
holdoutset = LandmarkDataset(X_holdout, y_holdout, transform = valid_transforms)
holdoutloader = DataLoader(holdoutset, batch_size = 96, shuffle = True, drop_last = True)


holdout_loss = 0
holdout_accuracy = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(holdoutloader):

        if batch_idx % 1000 == 0:
            print("Holdout Batch: " + str(batch_idx) + " from: " + str(len(holdoutloader)))

        best_model.eval()

        images = images.to(device)
        labels = labels.to(device)

        output = best_model(images.float())
        holdout_loss += criterion(output, labels.long())

        top_class = torch.argmax(output, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        holdout_accuracy += torch.mean(equals.type(torch.FloatTensor))

print("Final Loss on Holdout Set", holdout_loss.item() / len(holdoutloader))
print("Holdout Accuracy on Holdout Set", holdout_accuracy.item() / len(holdoutloader))

results = {

    "train_losses": train_losses,
    "train_accuracies": train_accuracies,

    "valid_losses": valid_losses,
    "valid_accuracies": valid_accuracies,

    "holdout_loss" : holdout_loss.item() / len(holdoutloader),
    "holdout_accuracy": holdout_accuracy.item() / len(holdoutloader)
}

print("Results")
print(results)

result_pickle = open(name + str(best_accuracy.item()) + ".pickle","wb")
pickle.dump(results, result_pickle)
result_pickle.close()
