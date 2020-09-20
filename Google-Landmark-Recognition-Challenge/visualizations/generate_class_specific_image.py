import torch
import torch.nn as nn

from torch.optim import Adam, SGD
from torchvision import models, transforms

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ClassSpecificImageGeneration():
    def __init__(self, path_to_pretrained_model, path_to_results, target_class, model_type):

        self.path_to_pretrained_model = path_to_pretrained_model     
        self.path_to_results = path_to_results
        self.target_class = target_class
        self.model_type = model_type   

        assert self.model_type in ["resnet18", "vgg16"], "Not Expected Model"

        self.model = self.load_pretrained_model()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))

        self.model.eval()


    def load_pretrained_model(self):

        if self.model_type == "resnet18":

            model = models.resnet18(pretrained = False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)

        else:

            model = models.vgg16(pretrained=False)
            model.classifier[3].out_features = 256
            model.classifier[6].in_features  = 256
            model.classifier[6].out_features = 10

            for param in model.classifier.parameters():
                 param.requires_grad = True

        model.load_state_dict(torch.load(self.path_to_pretrained_model))

        # model = models.alexnet(pretrained=True)

        return model


    def prepare_input(self):

        pil_image = Image.fromarray(self.created_image)

        input_image = self.transform(pil_image)

        input_image.unsqueeze_(0)

        input_image = torch.autograd.Variable(input_image, requires_grad=True)

        return input_image


    def reverse_image(self, image):
        image = image[0]
        image = image.detach().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        if np.max(image) <= 1:
            image = (image * 255).astype(np.uint8)

        return image

    def generate(self, learning_rate, num_iterations = 150):

        for iteration in range(1, num_iterations):

            input_image = self.prepare_input()

            optimizer = Adam([input_image], lr = learning_rate)

            output = self.model(input_image)

            class_loss = -output[0][self.target_class]

            print("Image Loss:" + str(class_loss.item()))

            self.model.zero_grad()

            class_loss.backward()

            optimizer.step()

            self.created_image = self.reverse_image(input_image)
            
            if iteration % 10 == 0 or iteration == num_iterations-1:
                model = "_Model_" + self.model_type 
                label = "_Class_" + str(self.target_class)
                index = "_Iteration_" + str(iteration)
                image_name = "ClassSpecificImageGeneration" + model + label + index + ".jpg"
                self.save_image(image_name)


    def save_image(self, file_name, display = False):

        path_to_save = os.path.join(self.path_to_results, file_name)

        image = self.created_image

        if np.max(image) <= 1:
            image = (image * 255).astype(np.uint8)

        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        image = Image.fromarray(image)

        image.save(path_to_save)

        if display:
            plt.imshow(image)
            plt.show()



def main():
    path_to_models  = "pretrained_models/"
    path_to_results = "results/"
    target_class    = 3

    model_type = "vgg16"

    models = [model for model in sorted(glob.glob(path_to_models + "*"))]

    # models[0] = resnet18 and models[1] = vgg16

    if model_type == "resnet18":
        path_to_pretrained_checkpoint = models[0]
    else:
        path_to_pretrained_checkpoint = models[1]


    image_generator = ClassSpecificImageGeneration(path_to_pretrained_checkpoint, path_to_results, target_class, model_type)

    image_generator = image_generator.generate(learning_rate = 0.6, num_iterations = 500)




if __name__ == "__main__":
    main()