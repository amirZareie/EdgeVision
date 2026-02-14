# Set up the Raspberry Pi, Group A and B
When starting this project, it’s best to have a fresh Raspberry Pi OS install to ensure that everything is up to date and there are no unforeseen conflicts on the system. You can use Raspberry Pi Imager to flash an SD card with a fresh OS: make sure to select the 64-bit Raspberry Pi OS Bookworm version. For more details on how to install a new OS on your Raspberry Pi, please follow their installation guide linked here.


Make sure to use the latest version of 64-bit Raspberry Pi OS. If you run into errors during this guide, it’s best to start fresh with a new install of the OS.
<img width="2344" height="1657" alt="image" src="https://github.com/user-attachments/assets/75275211-f804-474d-b503-ddb6ad5a6600" />


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


# Install Label Studio, Group B

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




