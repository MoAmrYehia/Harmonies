# Harmonies

<p align="center">
    <img src="https://raw.githubusercontent.com/MoAmrYehia/Harmonies/master/res/Harmonies_Logo.jpeg">
</p>

Harmonies is an online photo editor that aims to simplify the process of editing photos. Now, you can use the same advanced tools from photoshop by drag and drop easily. We take advantage of the capabilities of computer vision to help our users with edit photos in an appropriate way. Our web application is powerd by three different Computer Vision (CV) models. We submitted Harmonies to the [PyTorch Summer Hackathon 2020](https://devpost.com/software/pi-ke4nfz).

---

## Image Colorization
Generally, it will take you around 30 minutes to colorize (add color to) a black and white photo, but as you use Harmonies. You will bring new life to old photos by automatically colorizing them using the capabilities of Computer Vision! 

### State of the art
In this part we reimplemented the [Colorful Image Colorization](https://arxiv.org/abs/1603.08511v5) paper using PyTorch for images auto colorization

### Prerequisites
To install all dependencies run:

```Shell
cd image colorization
pip install -r requirements.txt
```
#### Running

```python
python app.py 
```

#### Request
Then send the following JSON request on `localhost:5000` 

```json
{
        "img": "data:image/jpeg;base64,/9j/2wCEAAgGBgcGBQ..."
}
```

#### Response
The response text will be the base64 representation of the masked colorized image.

#### Output
This is a real output using our model!

<p align="center">
    <img src="https://raw.githubusercontent.com/MoAmrYehia/pytorch-hackathon/master/res/2.jpg?token=AJUWNRZJ5DWPQGPZHISMHRS7HE6GC"
        width = "250" 
        height= "250">
    <img src="https://raw.githubusercontent.com/MoAmrYehia/pytorch-hackathon/master/res/1.jpg?token=AJUWNR2DEYR7FHLPZEVHS2K7HE6D2"
        width = "250" 
        height= "250">
</p>

---

## Image Segmentation
The pen tool is the most powerful tool in Illustrator and Photoshop. It allows you to define your own anchor points to extract elements from your image. Now you don't need all of this hard work, with Harmonies you can easily extract your image from the background and add it to another photo! 

#### State of the art
In this part we reimplemented this [paper](https://arxiv.org/abs/1706.05587v3) using PyTorch for auto-cropping a person from an image. We used the same concept of image segmentation and instead of adding masks, we return a PNG photo.

#### Implementation details
[detectron2](https://detectron2.readthedocs.io/) is a pytorch based, easy to use library for image segmentation.  

By feeding an image to detectron2 it produces a bunch of useful outputs, we are interested in two of them.
`output['instances'].pred_classes` and `output['instances'].pred_masks`.  
The first is an array of class labels telling us which object is being segmented (class 0 => person).  
The second is an array of bit masks that segment the respective objects in `pred_classes`.  
By choosing the bit masks corresponding to persons (class 0), converting them to `uint8`, summing them over the last axis, capping the values between 0 and 1 and multiplying the matrix by 255, we get the alpha layer of the output image.  
Then we just stack this layer with the input RGB image and we get a png image where only people are visible.

> **NOTE**  
> If you're running on a GPU, then the line `cfg.MODEL.DEVICE = 'cpu'` should be removed to reduce latency, however, if, like us, you are going to deploy this app to azure app service, this line is important.

#### Prerequisites
To install all dependencies run:
```
cd image-segmentation
pip install -r requirements.txt
```

#### Running

```python
python app.py 
```
#### Request
Then send the following JSON request on `localhost:5000` 

```json
{
        "img": "data:image/jpeg;base64,/9j/2wCEAAgGBgcGBQ..."
}
```

#### Response
The response text will be the base64 representation of the masked PNG image.


#### Output
This is a real output using our model!

<p align="center">
    <img src="https://raw.githubusercontent.com/MoAmrYehia/pytorch-hackathon/master/res/4.png"
        width = "250" 
        height= "250">
    <img src="https://raw.githubusercontent.com/MoAmrYehia/pytorch-hackathon/master/res/5.png"
        width = "250" 
        height= "250">
</p>

---

## Deep Painterly Harmonization
Copying an element from a photo and pasting it into a painting is a challenging task. Applying photo compositing techniques in this context yields subpar results that look like a collage. We introduce a technique to adjust the parameters of the transfer depending on the painting. For adding Harmonies to the painting and give a sense of uniqueness!.

#### State of the art
In this part we reimplemented this [paper](https://arxiv.org/abs/1804.03189) using PyTorch to add harmonies to the adjusted element. 

#### Prerequisites

#### Output
This is a real output using our model!

<p align="center">
    <img src="https://raw.githubusercontent.com/MoAmrYehia/pytorch-hackathon/master/res/3.jpg"
        width = "250" 
        height= "250">
</p>
