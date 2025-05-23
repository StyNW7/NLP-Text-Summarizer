# NLP ACLSum Text Summarizer - Final Project

## Simple Guide to run the Project

### 📌 Clone the repository
To clone the repo, please run this command:

```sh
git clone https://github.com/StyNW7/NLP-Text-Summarizer.git
```

### 📌 Creating Virtual Environment
To create virtual environment, please run this command:

```sh
# For Python 3
python -m venv venv
```

This command will create folder named `venv` to store all python dependencies for the project.

### 📌 Activate Virtual Environment
#### 🔹 Windows (Command Prompt)
```sh
venv\Scripts\activate
```

#### 🔹 Windows (PowerShell)
```sh
venv\Scripts\Activate.ps1
```

> ⚠️ If there is a warning in the Powershell, please run this command first.
> ```sh
> Set-ExecutionPolicy Unrestricted -Scope Process
> ```

### 🔹 macOS/Linux
```sh
source venv/bin/activate
```

If the activation succeed there is `(venv)` in the terminal.

### 📌 Install Dependencies
To install all the dependencies required, please run this command:

```sh
pip install -r requirements.txt
```

### 📌 Deactivate Virtual Environment
Run this command to deactivate the Virtual Environment:

```sh
deactivate
```

> [!TIP]
> 
> Setup the virtual environment and dependencies installation in the root folder

### 📌 Run the Project

```sh
# Go to the Final Project Scripts directory
cd Final-Project/scripts

# Preprocessing the dataset
python preprocess.py

# Training four models
python updated_train.py --model t5-small
python updated_train.py --model facebook/bart-base
python updated_train.py --model google/pegasus-xsum
python updated_train.py --model allenai/led-base-16384

# Evaluating the models
python evaluates_model.py

# Visualize the evaluation result
python visualize.py
```

> [!TIP]
> 
> Make sure to run the project inside (root)/Final-Project/scripts directory

#### Hopefully, this documentation and guide are helpful for learning and running the NLP project!

<!-- Owner -->

### Owner

This Repository is created by:
- Stanley Nathanael Wijaya - 2702217125

<code> Striving for Excellence ❤️‍🔥❤️‍🔥 </code>