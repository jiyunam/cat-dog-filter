import torchvision
import numpy as np
import matplotlib.pyplot as plt

from util import *

# functions to show an image
def imshow(img, class_name):
    """ For displaying the image from PyTorch Tensor
     Args:
         img: PyTorch image tensor. Note that the colour channel is the first dimension
    """
    img = img / 2 + 0.5     # Convert back to [0,1] range
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.figure()
    plt.title("Class: {}".format(class_name))
    plt.imsave("example_{}.png".format(class_name), npimg)

    return

def main():
    ########################################################################
    # Loads the configuration for the experiment from the configuration file
    config, learning_rate, batch_size, num_epochs, target_classes = load_config('configuration.json')

    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    train_loader, val_loader, test_loader, classes = get_data_loader(target_classes, batch_size)

    ########################################################################
    # We can visualize a batch of training data
    # First, get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # For each target class, plot their images
    for i in range(len(target_classes)):
        class_name = target_classes[i]
        print('Class: {}'.format(class_name))
        class_indices = labels == classes.index(target_classes[i])
        class_images = images[class_indices]
        # Save sample images
        imshow(torchvision.utils.make_grid(class_images), class_name)

if __name__ == '__main__':
    main()
