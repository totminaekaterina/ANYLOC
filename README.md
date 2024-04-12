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

Then, *descriptor bindings to clusters* were calculated for each image. To enhance visualisation, we have included a demonstration of the colours of each cluster in Fig.2. 

![colours](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/1.png)

Additionally, we have provided images in Fig.3 that display *the bindings of descriptors to clusters for each image*.

![descriptors](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/2.png)

Finally, Fig.4 displays *the results of the analysis*.

![results](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/3.png)

Selection of *specific clusters*, comprising 40% of the text, and an image comprising 60% (Fig. 5) was the final step.

![specific clusters](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/4.png)

---

The obtained results include:
1. Cluster centers, represented as coloured squares, which reflect the characteristics of various clusters.
2. Descriptors are linked to clusters in images, aiding in the understanding of the grouping of object characteristics into different clusters.
3. Images show descriptor bindings to clusters for each image.

---

### STAGE 3. ShallowMLP model training

To train the model, an approach based on building a *ShallowMLP* model. The purpose of this approach is to classify the room in the image using pre-selected clusters, which allows you to determine the type of room in the photo.

#### Performance evaluation

To evaluate the performance of the model, the compute_recalls function is used, which calculates the **R@1**, **R@5** and **R@10** indicators. These indicators represent the percentage of correct predictions for the one, five and ten most likely classes, respectively. This approach was presented by the authors in the process of training their model and is used to assess the accuracy of the classification.

#### Model Training

At this stage, iterations are performed by training epochs, the model is trained, its performance is evaluated on a validation dataset, the model is saved, and t-SNE plots are visualised to analyse the distribution of the data in space (Fig. 6).

![specific clusters](https://github.com/totminaekaterina/Anyloc-room-classification/blob/main/imgs/4.png)






