"""
Train and validate

This file contains the main training and validation loop of the chosen model
(FCN-8) for the city-scapes dataset. 
This file also contains all necessary functions for data pre-processing
and displaying results. 
"""

# %% Imports
from FCN8_model import FCN_8
from torchvision.datasets import Cityscapes
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# %% Show examples of data 
def show_data(data, class_names, colors, num_rows = 5):
    """
    Plots num_rows of subplots using images and masks stored in data. Figure with images is saved.

    Args:
        data: contains images and gt masks
        model: model can be defined to plot segmentation results
        num_rows (int): number of rows containing subplots
    """
    # Defining the plot
    fig = plt.figure(figsize=(15, 20))
    plt.subplots_adjust(hspace=0.35)
    plt.suptitle("Example images and their ground truth masks", fontsize= 18)
    counter = 1

    # Plotting num_rows of subplots
    for i in range(num_rows):
        img, mask = data[i]
        
        img = img.permute(1, 2, 0)
        img = renormalize_image(img.numpy())
        mask = mask.permute(1, 2, 0)
        mask = encode_mask(mask*255)
    
        unique_classes_gt = np.unique(mask)
        unique_classes_gt = np.delete(unique_classes_gt, np.where(unique_classes_gt == 255))
        classes_gt = [class_names[int(idx)] for idx in unique_classes_gt]
        
        # Split the classes into two for plot title
        split_index = len(classes_gt) // 2
        first_line = ', '.join(classes_gt[:split_index])   # First half of the list
        second_line = ', '.join(classes_gt[split_index:])  # Second half of the list
    
        mask_rgb = mask_to_rgb(mask.squeeze(), colors)
        fig.add_subplot(5, 2, counter)
        plt.imshow(img)
        plt.title('Image', fontsize = 9)
        plt.axis('off')

        ax = fig.add_subplot(5, 2, counter+1)
        plt.imshow(mask_rgb)
        plt.title(f'Ground truth mask:\n {first_line} \n {second_line}', fontsize = 9)
        plt.tight_layout()
        plt.axis('off')
          
        counter = counter + 2

    plt.savefig('train_data_fig.png')
    
# %% Show segmentation results
def show_results(data, predicted, class_names, colors, device="cpu", num_rows = 5):
    """
    Plots num_rows of subplots using images and masks stored in data. Figure with images is saved.

    Args:
        data: contains images and gt masks
        pred: predicted masks
        num_rows (int): number of rows containing subplots
    """
    # Defining the plot
    fig = plt.figure(figsize=(15, 20))
    plt.subplots_adjust(hspace=0.15)
    plt.suptitle("Example images, their ground truth, and predicted masks", fontsize= 18)
    counter = 1

    # Plotting num_rows of subplots
    for i in range(num_rows):
        img, mask = data[i]
        img, mask = img.to(device), mask.to(device)
        img = torch.unsqueeze(img, 0)
        
        pred = predicted[i]
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, 1)
        mask = encode_mask(mask * 255)

        img = img.squeeze()
        img = img.permute(1, 2, 0)
        img = img.cpu()
        pred = pred.cpu()
        mask = mask.cpu()
        img = renormalize_image(img.numpy())
        mask = mask.permute(1, 2, 0)
        
        unique_classes_pred = np.unique(pred)
        unique_classes_gt = np.unique(mask)
        
        unique_classes_pred = np.delete(unique_classes_pred, np.where(unique_classes_pred == 255))
        unique_classes_gt = np.delete(unique_classes_gt, np.where(unique_classes_gt == 255))
        
        print(f"Unique predicted classes: {unique_classes_pred}")
                                      
        classes_gt = [class_names[int(idx)] for idx in unique_classes_gt]
        classes_pred = [class_names[int(idx)] for idx in unique_classes_pred]
        
        # Split the classes into two for plotting
        split_index_gt = len(classes_gt) // 2
        first_line_gt = ', '.join(classes_gt[:split_index_gt])   # First half of the list
        second_line_gt = ', '.join(classes_gt[split_index_gt:])  # Second half of the list
        
        split_index_p = len(classes_pred) // 2
        first_line_p = ', '.join(classes_pred[:split_index_p])   # First half of the list
        second_line_p = ', '.join(classes_pred[split_index_p:])  # Second half of the list
        
        mask_rgb = mask_to_rgb(mask.squeeze(), colors)
        pred_rgb = mask_to_rgb(pred.squeeze(), colors)

        fig.add_subplot(5, 3, counter)
        plt.imshow(img)
        plt.title('Image', fontsize = 9)
        plt.axis('off')

        ax2 = fig.add_subplot(5, 3, counter + 1)
        plt.imshow(mask_rgb)
        plt.title(f'Ground truth mask:\n {first_line_gt} \n {second_line_gt}', fontsize = 5)
        plt.axis('off')
      
        ax3 = fig.add_subplot(5, 3, counter + 2)
        plt.imshow(pred_rgb)
        plt.title(f'Predicted mask:\n {first_line_p} \n {second_line_p}', fontsize = 5)
        plt.axis('off')
        
        plt.tight_layout()
      
        counter = counter + 3

    plt.savefig('results_fig.png')

# %% Renormalize image for displaying
def renormalize_image(image):
    """
    Renormalizes the image to its original range.

    Args:
        image (numpy.ndarray): Image tensor to renormalize.

    Returns:
        numpy.ndarray: Renormalized image tensor.
    """
    mean = [0.287, 0.325, 0.284]
    std = [0.176, 0.181, 0.178]
    renormalized_image = image * std + mean
    
    renormalized_image -= renormalized_image.min(axis=(0,1), keepdims=True)
    renormalized_image /= renormalized_image.max(axis=(0,1), keepdims=True)
    
    return renormalized_image

# %% Calculate data meand and std
def calculate_mean_std(data):
    """
    Solves the channel-wise mean and standard deviation of a dataset.

    Parameters:
        dataset (numpy.ndarray): The dataloader for data whose standard deviation is solved.

    Returns:
        total_mean, total_std: Mean and standard deviation for each channel.
    """
    total_mean = []
    total_std = []
    
    for image, mask in data:
        image = image.numpy()
        
        batch_mean = np.mean(image, axis=(1,2))
        total_mean.append(batch_mean)
    
        batch_std = np.std(image, axis=(1,2))
        total_std.append(batch_std)

    total_mean = np.array(total_mean).mean(axis=0)
    total_std = np.array(total_std).mean(axis=0)

    return total_mean, total_std

# %% Convert mask to rgb
def mask_to_rgb(mask, class_to_color):
    """
    Converts a numpy mask with multiple classes indicated by integers to a color RGB mask.

    Parameters:
        mask (numpy.ndarray): The input mask where each integer represents a class.
        class_to_color (dict): A dictionary mapping class integers to RGB color tuples.

    Returns:
        numpy.ndarray: RGB mask where each pixel is represented as an RGB tuple.
    """
    # Get dimensions of the input mask
    height, width = mask.shape

    # Initialize an empty RGB mask
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate over each class and assign corresponding RGB color
    for class_idx, color in class_to_color.items():
        # Mask pixels belonging to the current class
        class_pixels = mask == class_idx
        # Assign RGB color to the corresponding pixels
        rgb_mask[class_pixels] = color

    return rgb_mask

# %% Encode mask
def encode_mask(mask):
    """
    Converts a mask with class labels that should be ignored into a mask with only valid labels.

    Parameters:
        mask (numpy.ndarray): The input mask with unwanted labels.
        
    Returns:
        numpy.ndarray: RGB mask where each pixel is represented by values between 0-19.
    """
    # Defining labels we want to ignore/keep
    ignore_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    real_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    # Replacing lables in mask to ones we want to keep
    for ignore_value in ignore_labels:
        mask[mask == ignore_value] = 255
    for value in real_labels:
        mask[mask == value] = real_labels.index(value)
        
    return mask

# %% Run main program
def main():
    
    data_path = '/city-scapes/data' # Add a location that works for you
    
    # Data loading and applying transformations
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 64), transforms.InterpolationMode.NEAREST),
    transforms.Normalize((0.287, 0.325, 0.284), (0.176, 0.181, 0.178))
    ])
    
    target_transforms = transforms.Compose([
    transforms.Resize((32, 64), transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
    ])

    dataset = Cityscapes(data_path, split = 'train', mode = 'fine', target_type = 'semantic', transform = transform, target_transform = target_transforms)
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [2376, 594, 5], generator = torch.Generator().manual_seed(1))

    # Visualizing example images from train_data
    class_names = ['road', 'sidewalk','building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                   'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    colors = {
        0: (128, 64, 128),
        1: (244, 35, 232),
        2: (70, 70, 70),  
        3: (102, 102, 156),   
        4: (190, 153, 153), 
        5: (153, 153, 153),
        6: (250, 170, 30),
        7: (220, 220, 0),
        8: (107, 142, 35),
        9: (152, 251, 152),
        10: (70, 130, 180),
        11: (220, 20, 60),
        12: (255, 0, 0),
        13: (0, 0, 142),
        14: (0, 0, 70),
        15: (0, 60, 100),
        16: (0, 80, 100),
        17: (0, 0, 230),
        18: (199, 11, 32),
        255: (0, 0, 0)
        }
    
    show_data(test_data, class_names, colors)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Defining model and training parameters
    model = FCN_8(3, 34).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = 255)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    
    num_epochs = 20
    epochs = range(1, num_epochs+1)
    train_losses = []
    val_losses = []
    
    # Training + validation loop
    for epoch in range(num_epochs):
        model.train()
        tr_loss = 0.0
        
        # Training loop
        train_time_start = time.time()
        for inputs, gt_masks in tqdm(train_data, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, gt_masks = inputs.to(device), gt_masks.to(device)
            inputs = torch.unsqueeze(inputs, 0)
            optimizer.zero_grad()
            
            tr_pred = model(inputs)
            gt_masks = gt_masks * 255  
            gt_masks = encode_mask(gt_masks)
            
            loss = criterion(tr_pred, gt_masks.long())
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            tr_loss = tr_loss / len(train_data)
            
        print(f'{time.time()-train_time_start:.2f} seconds for training')
        
        # Validation loop
        model.eval()
        val_time_start = time.time()
        val_loss = 0.0

        with torch.no_grad():
            for val_inputs, val_masks in val_data:
                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)
                val_inputs = torch.unsqueeze(val_inputs, 0)
            
                val_pred = model(val_inputs)
            
                val_masks = val_masks * 255  
                val_masks = encode_mask(val_masks)
                val_loss = criterion(val_pred, val_masks.long())
            
                val_loss += loss.item()
                val_loss = val_loss / len(val_data)
                
        print(f'{time.time()-val_time_start:.2f} seconds for validation')
        train_losses.append(tr_loss)
        val_losses.append(val_loss.tolist())
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {tr_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Visualizing training and validation losses
    fig = plt.figure(figsize=(8, 6))
    plt.title("Train and validation loss", fontsize = 12)
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['Train loss', 'Validation loss'])

    plt.savefig('loss_curve.png')
    
    # Save model weights
    torch.save(model.state_dict(), 'model_weights.pth')
    
    # Visualizing results
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(test_data):
            inputs = inputs.to(device)
            inputs = torch.unsqueeze(inputs, 0)           
            pred = model(inputs)
            pred = encode_mask(255 * pred)
            predictions.append(pred)
            
    show_results(test_data, predictions, class_names, colors)

    pass

if __name__ == "__main__":
    main()