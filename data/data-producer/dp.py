#! Data Producer
#! Author: Waasiq Masood (https://github.com/waasiq) 
#! 3 - Feb - 2022
import random
import csv

rows , columns = (5000,6)
dataset = [[0 for i in range(columns)] for j in range(rows)]
header = [ 'pH' , 'Turbidity', 'Chloramine', 'Dissolved Oxygen', 'Solids','Potability']

'''
#? Parameters to consider :
#*      ph: 6.5 - 8.5
#*      Turbidity: 1 NTU - 5 NTu 
#*      Chloramine: <= 4 mg/L
#*      Dissolved Oxygen: 6.5-8 mg/L
#*      Solids: <= 1200 mg/L

#?      -- Temperature: 25 Centigrade
'''

def makeDataSet(): 
    for x in range(5000):
        num = random.randint(0,4)
        for y in range(6):
            if (x < 4500):
                if (y == 0):
                    dataset[x][y] = round(random.uniform(6.5,8.5), 2)
                elif (y == 1):
                    dataset[x][y] = round(random.uniform(1,5), 2)
                elif (y == 2):
                    dataset[x][y] = round(random.uniform(0,4), 2)
                elif (y == 3):
                    dataset[x][y] = round(random.uniform(6.5,8), 2)
                elif (y == 4):
                    dataset[x][y] = round(random.uniform(0,1200), 2)
                elif (y == 5): 
                    dataset[x][y] = 'TRUE'
            else: 
                if (y == 5): 
                    dataset[x][y] = 'FALSE'             
                elif (y == 0):
                    dataset[x][y] = round(random.uniform(6.5,8.5), 2) if num != 0 else round(random.uniform(9.5,14), 2)            
                elif (y == 1):
                    dataset[x][y] = round(random.uniform(1,5), 2) if num != 1 else round(random.uniform(7,100), 2)             
                elif (y == 2):
                    dataset[x][y] = round(random.uniform(0,4), 2) if num != 2 else round(random.uniform(6,100), 2)                   
                elif (y == 3):
                    dataset[x][y] = round(random.uniform(6.5,8), 2)  if num != 3 else round(random.uniform(9, 100), 2)              
                elif (y == 4):
                    dataset[x][y] = round(random.uniform(0,1200), 2)  if num != 4 else round(random.uniform(1300,2500), 2)
                          

        
def export():
    with open("../custom-dataset.csv", "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dataset)


def main(): 
    makeDataSet()
    export()

if __name__ == '__main__': 
    main()