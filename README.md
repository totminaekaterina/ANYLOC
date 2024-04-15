# AnyLoc room classification task

## The main aspects of AnyLoc

[AnyLoc](https://anyloc.github.io/assets/AnyLoc.pdf) is a method for visual location recognition (VPR) that is versatile and effective across diverse datasets, locations, times, and perspectives. Its aim is to provide a universal VPR solution by incorporating self-monitoring features like *DINOv2* and unsupervised aggregation methods such as *VLAD* and *GeM*, ensuring robust performance.

The method creates a dictionary for aggregation by using vocabulary to describe the semantic properties of local features in different environments. This involves three steps:

1. Creating a **global dictionary** based on representative places and attributes.

2. Developing a **dictionary specific** to a particular map.

3. Training a **custom dictionary**.

Global descriptors from *DINOv2-GeM* are used along with *PCA projection* to achieve the following objectives:

1. Utilising **global dictionary-independent descriptors**.

2. Identifying different **domains** in a hidden space without supervision through PCA projection.

## Setting room classification task

For this task, a method was developed to classify room types based on provided images, taking into account the approach implemented in the ANYLOC project. 

*The following experiments are presented to address this issue.*

## Experiment №1

A dataset of 40 photos was presented, featuring four different room types: *Kitchen*, *Bedroom*, *Living Room*, and *Bathroom*. Each category contained **10 photos**.

The solution to this problem can be presented in successive stages:

1. The uploaded and preprocessed images are processed using DINOv2 vision transformers to extract quality features. The data is processed in .npy format for Global Descriptor Generation.
2. Creating clusters of assignments for the VLAD disordered aggregation method
3. Selection of specific (specific) clusters.
4. Training the model to perform the classification task on the received data.
6. Classification of the previously processed user image.

### STAGE 1. DINOv2 Descriptors

The initial step involved loading and processing the image (Fig.1). This included extracting DINO features and generating global VLAD descriptors for each photo. The results were saved in .npy format files to store data arrays in NumPy format. These preparations were made for their future use.

![living_room_example](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/lr_4.jpg)

### STAGE 2. Using the VLAD method to create assignment clusters and highlight the specific clusters

The Visual Localization and Detection (VLAD) method was used for image analysis at this stage, as part of the process of searching for objects in images.

The VLAD method required loading cluster centers from the cache file. And residual vectors were generated for each image using a DINOv2 extractor. 

Additionally, we have provided images that display *the bindings of descriptors to clusters for each image*.

Finally, we got *the results of the analysis*.

Selection of *specific clusters*, comprising 40% of the text, and an image comprising 60% (Fig. 2) was the final step.

![specific clusters](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/4.png)

---

The obtained results include:
1. Cluster centers, represented as coloured squares, which reflect the characteristics of various clusters.
2. Descriptors are linked to clusters in images, aiding in the understanding of the grouping of object characteristics into different clusters.
3. Images show descriptor bindings to clusters for each image.

---

### STAGE 3. ShallowMLP model training and classification

To train the model, an approach based on building a *ShallowMLP* model. The purpose of this approach is to classify the room in the image using pre-selected clusters, which allows you to determine the type of room in the photo.

#### Performance evaluation

To evaluate the performance of the model, the compute_recalls function is used, which calculates the **R@1**, **R@5** and **R@10** indicators. These indicators represent the percentage of correct predictions for the one, five and ten most likely classes, respectively. This approach was presented by the authors in the process of training their model and is used to assess the accuracy of the classification.

#### Model Training

At this stage, iterations are performed by training epochs, the model is trained, its performance is evaluated on a validation dataset, the model is saved, and t-SNE plots are visualised to analyse the distribution of the data in space (Fig. 3).

![t-sne-1](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/t_sne_1.png)

#### Training results

After 10 epochs of training, the model showed the following metric estimates (after the **8th epoch** of training, the model began **re-training**):

| Epoch | Metric | Result |
|-------|--------|--------|
| 1     | loss   | 8.56   |
|       | accuracy | 0.35  |
|       | [R@1, R@5, R@10] | [0.025, 0.1, 0.125] |
| 5     | loss   | 1.89   |
|       | accuracy | 0.95  |
|       | [R@1, R@5, R@10] | [0.025, 0.075, 0.15] |
| 8     | loss   | 0.0001 |
|       | accuracy | 1.0   |
|       | [R@1, R@5, R@10] | [0.025, 0.075, 0.175] |
| 10    | loss   | 2.7    |
|       | accuracy | 1.0   |
|       | [R@1, R@5, R@10] | [0.025, 0.075, 0.175] |



#### Classification stage

The images of the rooms were transformed using DINO v2 and VLAD, and specific clusters were allocated to them, as well as in the training sample. They were then classified using our pre-trained ShallowMLP model, which outputs the predicted room class and the probability of this prediction.

As a result, the accuracy of the classification of 10 user images was only 40%, with only 4 out of 10 images being classified correctly.


## Experiment №2

The experiments were conducted in two main areas: *dataset* and *model setup*.

### Dataset

The dataset was expanded to include 250 photos from 5 different rooms, with 50 photos for each class. This expansion improves the model's generalizing ability, reduces the risk of overfitting, reduces learning variability, and provides reliable statistics for analyzing the results.

### Model

Changes were made to the model setup, including reducing the packet size, increasing the number of training epochs, using the Adam method for optimization, employing the cross-entropy loss function, and normalizing the data. However, although T-SNE showed better convergence (Fig.3) , there was no improvement in the R@1, R@5, and R@10 metrics.

![t-sne-2](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/t_sne_2.png)

## Experiment №3

The experiment utilized the **`vt_base_patch 32_224`** model of VIT class transformers with an attention mechanism to classify images. 
1. The number of *training epochs* was increased to 100, and a test dataset was added.
2. The *test dataset* comprised 25 photos (5 for each room), and the *training dataset* comprised 225 photos (45 for each room).
3. The *loss graph* indicated the start of *retraining after the 10th epoch* (Fig.4).
4. The *number of clusters* was increased to *32* for more detailed image analysis.

![loss_3](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/loss_3.png)

However, these changes did not enhance the model's classification ability, so the experiments were continued.

## Experiment №4

The fourth experiment utilized the `vt_base_patch 32_224` model. However, clusters were not allocated using DINO v2 and VLAD in the training and test datasets. The charts for loss and t-SNE are presented below (Fig.5) along with the classification probability results for different images. The model has demonstrated good classification abilities, despite the lack of grouping in the patterns.

![loss_5](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/loss_5.png)

The classified classes are as follows: 'Kitchen' (0), 'Living_room' (1) (Fig.6-7), 'Bedroom' (2), 'Bathroom' (3), and 'Wardrobe' (4).

![lr_ph](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/lr_4.jpg)
![lr_res](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/lr_res.png)

| Image № | Image Class | Predicted Image Class | Propapility| Right/False |
|:------------|:-----------:|:-----------:|:-----------:|:-----------:|
| 1 | Wardrobe |  Wardrobe |  0.52 | **Right** |
| 2 | Bathrooom | Bathrooom | 0.54 | **Right** |
| 3 | Bedroom |  Bedroom| 0.44 | False |
| 4 | Kitchen | Kitchen | 0.37 | **Right** |
| 5 | Living_room | Living_room | 0.38 | **Right** |
| 6 | Wardrobe | Wardrobe | 0.56 | **Right** |
| 7 | Bedroom | Bedroom | 0.41 | **Right** |
| 8 | Bathrooom | Bathrooom | 0.57 | **Right** |
| 9 | Kitchen | Kitchen | 0.42 | **Right** |
| 10 | Living_room | Living_room | 0.33 | False |


### Results

The last 4th experiment demonstrated an improvement in the model's predictive ability. Specifically, the model correctly classified images into their respective classes with an accuracy rate of **80% out of 100%**. 

The 'Living_room' and 'Bedroom' classes displayed the poorest results, likely due to an imbalance in the dataset. The training sample for these classes included images from different layouts, which could have impacted the model's training. It is assumed that a larger and more balanced dataset will result in improved performance for these classes.

From this, we can conclude that for training purposes: *it is better to use the initial raw images that have been cropped, without first selecting clusters*.

## Conclusion

Based on the experiments conducted, the following conclusion can be drawn:
1.   When using the **ShallowMLP model**, *dataset size is a crucial criterion*. The model performed better with fewer data points (50 images) than with more (225 images).
2.   Additionally, **increasing the number of clusters** can enhance the model's learning ability, but it requires selecting the appropriate architecture. For instance, when the number of clusters was increased to **32**, we utilized the **`vt_base_patch 32_224`** model, which improved the classification ability. In contrast, the **ShallowMLP** model exhibited inferior results with the same number of clusters.
4. Finally, **highlighting clusters** in images can **interfere** with the model's learning. Pre-clustered images may inaccurately convey the general characteristics of classes. The model demonstrated better results without cluster allocation.

## Examples

The example folder's repository includes notebooks for the second and fourth experiments to enhance clarity.

