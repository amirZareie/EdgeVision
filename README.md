### Step 1 - Set up the Raspberry Pi, Group A and B
When starting this project, it’s best to have a fresh Raspberry Pi OS install to ensure that everything is up to date and there are no unforeseen conflicts on the system. You can use Raspberry Pi Imager to flash an SD card with a fresh OS: make sure to select the 64-bit Raspberry Pi OS Bookworm version. For more details on how to install a new OS on your Raspberry Pi, please follow their installation guide linked here.


Make sure to use the latest version of 64-bit Raspberry Pi OS. If you run into errors during this guide, it’s best to start fresh with a new install of the OS.
<img width="2344" height="1657" alt="image" src="https://github.com/user-attachments/assets/75275211-f804-474d-b503-ddb6ad5a6600" />


Plug in a monitor, keyboard, and mouse into your Raspberry Pi and turn it on. Connect it to the internet over WiFi.

Step 1a - Update the Raspberry Pi
From the home screen, open a terminal and issue the following commands to make sure that your Raspberry Pi is up to date with its OS:


`sudo apt update && sudo apt upgrade -y`

The update could take a few minutes to complete.

### inference virtual environment, Group A and B
Next, let’s create a working directory to hold our models and code files in. Create a new directory called “EdgeVision” by issuing:

`mkdir ~/edgevision
cd ~/edgevision`

Now we need to create a virtual environment to install Python libraries in. Using a virtual environment allows us to avoid version conflicts with existing Python libraries on the Raspberry Pi OS. Create a virtual environment named “inference” and then activate it by issuing:

`python3 -m venv --system-site-packages inference
source inference/bin/activate`

When the environment is active, “(inference)” will appear before the path in the command prompt, as shown in the image below.

Next, let’s install the Ultralytics and NCNN libraries that will be used for running the YOLO models in NCNN format. Issue the following command:

`pip install ultralytics ncnn`
