import subprocess
import os

FILE_NAME = "submit.csv"
MESSAGE = ""

def submit_to_kaggle(dataframe):

    if os.path.exists("submit.csv"):
        os.remove("submit.csv")
    dataframe.to_csv("submit.csv", index=True)

    command = "kaggle competitions submit -c open-problems-single-cell-perturbations -f " + FILE_NAME + ' -m "' + MESSAGE + '"'
    print("Running " + command)
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output
    print("Output:\n", result.stdout)

    # Print any errors
    if result.stderr:
        print("Error:\n", result.stderr)
