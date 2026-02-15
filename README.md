# Set up the Raspberry Pi, Group A and B
Plug in a monitor, keyboard, and mouse into your Raspberry Pi and turn it on. Connect it to the internet over WiFi.

Step 1a - Update the Raspberry Pi
From the home screen, open a terminal and issue the following commands to make sure that your Raspberry Pi is up to date with its OS:


```bash
sudo apt update && sudo apt upgrade -y
```

The update could take a few minutes to complete.

# Set up inference virtual environment, Group A and B
Next, let’s create a working directory to hold our models and code files in. Create a new directory called “EdgeVision” by issuing:

```bash
mkdir ~/edgevision
cd ~/edgevision
```

Now we need to create a virtual environment to install Python libraries in. Using a virtual environment allows us to avoid version conflicts with existing Python libraries on the Raspberry Pi OS. Create a virtual environment named “inference” and then activate it by issuing:

```bash
python3 -m venv --system-site-packages inference
source inference/bin/activate
```

When the environment is active, “(inference)” will appear before the path in the command prompt, as shown in the image below.

Next, let’s install the Ultralytics and NCNN libraries that will be used for running the YOLO models in NCNN format. Issue the following command:

```bash 
pip install ultralytics ncnn
```


# Install and use Label Studio, Group B

This guide explains how to create a Python virtual environment named **`LabelStudioEnv`** on Raspberry Pi OS and install **Label Studio** inside it.


Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

Install Required Dependencies

```bash
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential
```

 Create a Virtual Environment

```bash
python3 -m venv LabelStudioEnv
```

Activate the Virtual Environment

```bash
source LabelStudioEnv/bin/activate
```

Your shell prompt should now show:

```
(LabelStudioEnv) $
```

Upgrade pip

```bash
pip install --upgrade pip
```


Install Label Studio

```bash
pip install label-studio
```

Launch Label Studio

```bash
label-studio
```

Open the printed URL (usually `http://localhost:8080`) in your browser to access the interface.
<img width="1627" height="920" alt="image" src="https://github.com/user-attachments/assets/47a13b27-e338-421a-aa9e-eac6766a7e21" />

click on sign up and create a dummy account for the application. 
this account remains on the raspberry pi as the lablestudio service is running locally. 

<img width="578" height="867" alt="Screenshot from 2026-02-15 05-52-21" src="https://github.com/user-attachments/assets/3026a0b7-116d-40a1-a56d-f1c4959673c0" />

sign in with the account that you created and then create new project. 
after logging in, create a new project. select a project name and import all images. please note if you have more that 100 images you need to upload them in two rounds. 
in "Labeling Setup", Select "Object Detection with Bounding Boxes"

<img width="516" height="346" alt="Screenshot from 2026-02-15 05-58-54" src="https://github.com/user-attachments/assets/9265a179-265a-4442-8aa5-12a7f09d5bad" />

Label all the pictures and then when all pictures are annotated, press Export button and select "YOLO with Images". the project will get downloaded as zip file. rename it to data.zip.


# Training Custom YOLO Model
Click below to acces a Colab notebook for training YOLO models. you need Label Studio project export for this notebook. 
https://colab.research.google.com/github/amirZareie/EdgeVision/blob/main/train_yolo_models.ipynb

# deploy trained model to Raspberry Pi 5
after downloading your custom yolo model you should have a my_model.zip file. 
now activate the inference environment that has been created ealier by running below command in terminal: 

```bash
source inference/bin/activate
```
unzip my_model.zip file to a directory, you should be able to see .pt model. given we want to run the model on raspberry pi which does have limited resources, we need to reformat pt (pyTorch) model to lighter ncnn one. 
this can be done with yolo export command as below: 

```bash
yolo export model=my_model.py format=ncnn
```
now you should have my_model_ncnn_model in your directory. 

to run the yolo model you can use yolo_detect.py script in this repository. 
first get the script file via below command: 

```bash
curl --ouput yolo_detect.py https://raw.githubusercontent.com/amirZareie/EdgeVision/refs/heads/main/yolo_detect.py
```
to run you can use below terminal command: 

```bash
python yolo_detect.py --model=my_model_ncnn_model --source=picamera0 --resolution=1920x768
```
the arguments for yolo_detect.py:
    --model: Path to a model file (e.g. my_model.pt). If the model isn't found, it will default to using yolov8s.pt.
    
    --source: Source to run inference on. The options are:
        Image file (example: test.jpg)
        Folder of images (example: my_images/test)
        Video file (example: testvid.mp4)
        Index of a connected USB camera (example: usb0)
        Index of a connected Picamera module for Raspberry Pi (example: picamera0)
        
    --thresh (optional): Minimum confidence threshold for displaying detected objects. Default value is 0.5 (example: 0.4)
    
    --resolution (optional): Resolution in WxH to display inference results at. If not specified, the program will match the source resolution. (example: 1280x720)
    
    --record (optional): Record a video of the results and save it as demo1.avi. (If using this option, the --resolution argument must also be specified

I also put another inference example that does google search for the title of object when user click on the object box. the instruction is the same just replace yolo_detect.py with yolo_detect_googleSearch.py 




