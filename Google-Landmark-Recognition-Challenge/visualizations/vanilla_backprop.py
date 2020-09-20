import torch
import torch.nn as nn
from torchvision import models, transforms

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class VanillaBackprop():
    def __init__(self, path_to_pretrained_model, path_to_image, path_to_results, model_type):
        self.path_to_image = path_to_image
        self.path_to_results = path_to_results
        self.path_to_pretrained_model = path_to_pretrained_model     

        self.model_type = model_type   

        assert self.model_type in ["resnet18", "vgg16"], "Not Expected Model"

        self.gradients = None

        self.model = self.load_pretrained_model()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.original_image, self.input_image, self.expected_target = self.prepare_input()

        self.model.eval()
        self.hook_layers()


    def prepare_input(self):

        # The label should be the name of the image

        original_image = Image.open(self.path_to_image)

        expected_target = int(self.path_to_image.split("/")[-1].split(".")[0].split("-")[-1])

        input_image = self.transform(original_image)

        input_image.unsqueeze_(0)

        input_image = torch.autograd.Variable(input_image, requires_grad=True)

        return original_image, input_image, expected_target


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

        return model


    def hook_layers(self):

        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        if self.model_type == "resnet18":
            first_layer = self.model.conv1
        else:
            first_layer = self.model.features[0]

        first_layer.register_backward_hook(hook_function)


    def generate_gradients(self):

        output = self.model(self.input_image)

        predicted_class = torch.argmax(output, dim = 1).item()

        print("Predicted label is " + str(predicted_class) + " and expected target is: " + str(self.expected_target))

        self.model.zero_grad()

        one_hot_output = torch.FloatTensor(1, output.shape[-1]).zero_()

        one_hot_output[0][self.expected_target] = 1

        output.backward(gradient = one_hot_output)

        self.gradients = self.gradients.data.numpy()[0]

        return self.gradients


    def save_gradient_image(self, file_name, normalize = True, saliency = False, display = True):

        path_to_save = os.path.join(self.path_to_results, file_name)

        image = self.gradients

        # For Grayscale
        if saliency:
            image = np.sum(np.abs(image), axis = 0)
            im_max = np.percentile(image, 99)
            im_min = np.min(image)
            image = (np.clip((image - im_min) / (im_max - im_min), 0, 1))
            image = np.expand_dims(image, axis = 0)

        if normalize:
            image = image - image.min()
            image /= image.max() - image.min()
        else:
            if self.model_type == "vgg16":
                image = np.clip(image, 0, 1)

        image = image.transpose(1, 2, 0)

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
    #Labelul este numele imaginii
    path_to_image   = "input_images/Class-3.jpg"
    path_to_models  = "pretrained_models/"
    path_to_results = "results/"

    normalize  = True 
    saliency   = True # For Grayscale
    model_type = "vgg16"

    models = [model for model in sorted(glob.glob(path_to_models + "*"))]

    # models[0] = resnet18 and models[1] = vgg16

    if model_type == "resnet18":
        path_to_pretrained_checkpoint = models[0]
    else:
        path_to_pretrained_checkpoint = models[1]

    model      = "_Model_" + model_type 
    label      = "_Class_" + path_to_image.split("/")[-1].split(".")[0].split("-")[-1]
    normalized = "_Normalize_" + str(normalize)
    saliece    = "_Saliency_" + str(saliency)

    image_name = "VanillaBackprop" + model + label + normalized + saliece + ".jpg"

    vanilla = VanillaBackprop(path_to_pretrained_checkpoint, path_to_image, path_to_results, model_type)

    vanilla_grads = vanilla.generate_gradients()

    vanilla.save_gradient_image(image_name, normalize = normalize, saliency = saliency)



if __name__ == "__main__":
    main()