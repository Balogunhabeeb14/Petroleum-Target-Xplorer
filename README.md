# Petroleum-Target-Xplorer

The data summary provides us with the metadata for the data. Herein, few observations found includes:

- Litho-logs comprise features {Spectral Gamma, GR, SP, Mud-log, Caliper}
- Porosity	comprises features {DT,	RHOB, CNL}
- RES	comprises features {LLD, LLS}
- Biostratigraphic Data
- Well-Tops/Formation Tops
- Reports	comprises features {Geologic,	Rock-Eval}
- Rock-Eval	comprises features {S1, S2, S3,	HI,	OI, VR,	TOC, Tmax}

These features were collected mostly for Kolmani river 2(KR2) and Kolmani river 3(KR3). To have a fair analysis, we will only include features that are available in both KR2 & KR3.

PS. Most of the Multivariate linear regression analysis was repeated to confirm the dependent variable, what was discovered was that, the TOC(measured) was the actual variable predicted across all analysis.

To replicate the analysis; help you set up and run the project on your local machine.

Prerequisites
Before you begin, ensure you have the following installed on your system:

Python 3.x
Git
Step 1: Clone the GitHub Repository
Start by cloning the repository to your local machine. Open a terminal or command prompt and run the following command:

bash
Copy code
git clone https://github.com/Balogunhabeeb14/Petroleum-Target-Xplorer.git
Navigate into the project directory:

bash
Copy code
cd Petroleum-Target-Xplorer


Step 2: Set Up Virtual Environment
It is recommended to use a virtual environment to manage dependencies for the project. You can create and activate a virtual environment using venv:

On Windows:
Create a virtual environment:

bash
Copy code
python -m venv venv
Activate the virtual environment:

bash
Copy code
venv\Scripts\activate

On macOS/Linux:
Create a virtual environment:

bash
Copy code
python3 -m venv venv
Activate the virtual environment:

bash
Copy code
source venv/bin/activate
After activating the virtual environment, your terminal should show (venv) at the beginning of each command line.

Step 3: Install the Required Dependencies
Once the virtual environment is activated, install the dependencies listed in the requirements.txt file:

bash
Copy code
pip install -r requirements.txt
This will ensure that all the necessary packages are installed for the project to run.

Step 4: Running the Project
After setting up the environment and installing the required packages, you can run the project. Here are some possible commands you might use to execute the project (customize based on your project’s structure):

bash
Copy code
python /Petroleum-Target-Xplorer/Second paper/Recent.ipynb
Or if there are specific scripts for running models or analyses:


Additional Notes
Make sure you have stated the proper data location 
You may need to adjust environment variables or configuration files before running the project.
