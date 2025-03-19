# nba-prediction

Running Instruction:-

open terminal on your device and then clone the directory

git clone https://github.com/vaghelahetvi/nba-prediction.git

then go to the directory with 
cd nba-prediction

then install all the requirements
pip install -r requirements.txt

now run the model trainer
python nba-player-prediction.py

now that you have your model ready use it on the test data to find out the missing player
python missing-player-predictor.py

to preview the result 
less Results/NBA_test_predictions.csv
press q to exit after viewing

to preview the yearly result 
less Results/NBA_test_yearly_accuracy.csv
press q to exit after viewing
