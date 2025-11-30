# DS-4002-Project-2
## We Hate Tickets - A ticket prediction model for Charlottesville VA
This project analyzes the Charlottesville Open Data Portal to train various models based on different time frames of tickets within the city. In the end, we arrived at one final model that had the greatest accuracy at **96.6%**, which was trained on the **2009-2013** time frame.

## Running
1) Base Installations
    - Install Python 3.10+
        - more information can be found at https://www.python.org/downloads/
2) Project Installations
    - Create python environment
        - macOS/Linux --> `python -m venv .venv && source .venv/bin/activate`
        - Windows --> `python -m venv .venv && .venv\Scripts\activate`
            - Note, if you are user a newer python it may be `python3 ...` for all commands
    - Install required packages
        - within the terminal, `pip install -r requirements.txt`
    - Register the Jupyter kernel (if using notebooks outside VS Code):
        - `pip install ipykernel`
        - `python -m ipykernel install --user --name=.venv`
        - alternatively, in VS code you may have to define your interpreter as the one located in your venv. 
3) Run
    - If you want to recreate all final files, within the base directory, run in order...
        1) `clean_parking_data.py` to re-create the cleaned data
        2) `create_models.py` to re-create the models on the train split
        3) `score_models.py` to re-score the models
    - View the output in `/OUTPUT/`, which includes the models as well as the performance in `model_performance.csv`.
