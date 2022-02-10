#! Data Producer
#! Author: Waasiq Masood (https://github.com/waasiq) 
#! 3 - Feb - 2022
import random
import csv

rows , columns = (5000,6)
dataset = [[0 for i in range(columns)] for j in range(rows)]
header = [ 'pH' , 'Turbidity', 'Chlorine', 'Dissolved Oxygen','Temperature' ,'Potability']

'''
#? Parameters to consider :
#*      ph: 6.5 - 8.5
#*      Turbidity: 1 NTU - 5 NTu 
#*      Chloramine: 0.5ppm - 1ppm
#*      Dissolved Oxygen: 6.5-8 mg/L
#*      Temperature: 0 C and 25 C and 50 C 
#?      Potability
'''

def makeDataSet(): 
    for x in range(5000):
        num = random.randint(0,4)
        temp = [0,25,50]
        tempRand = random.randint(0,2)
        for y in range(6):
            if (x%13 != 0):
                if (y == 0):
                    if temp[tempRand] == 0:
                        dataset[x][y] = round(random.uniform(6.5,8.5), 2)
                    if temp[tempRand] == 25:
                        dataset[x][y] = round(random.uniform(6.10,8.10), 2)
                    elif temp[tempRand] == 50:
                        dataset[x][y] = round(random.uniform(6.97,8.97), 2)                    
                elif (y == 1):
                    dataset[x][y] = round(random.uniform(1,5), 2)
                elif (y == 2):
                    dataset[x][y] = round(random.uniform(0.5,1), 2)
                elif (y == 3):
                    dataset[x][y] = round(random.uniform(6.5,8), 2)
                elif (y == 5): 
                    dataset[x][y] = 'TRUE'
            else: 
                if (y == 5): 
                    dataset[x][y] = 'FALSE'             
                elif (y == 0):
                    dataset[x][y] = round(random.uniform(6.5,8), 2) if num != 0 else round(random.uniform(9.5,14), 2)            
                elif (y == 1):
                    dataset[x][y] = round(random.uniform(1,5), 2) if num != 1 else round(random.uniform(7,20), 2)             
                elif (y == 2):
                    dataset[x][y] = round(random.uniform(0.5,1), 2) if num != 2 else round(random.uniform(3,20), 2)                   
                elif (y == 3):
                    dataset[x][y] = round(random.uniform(6.5,8), 2)  if num != 3 else round(random.uniform(9, 20), 2)              
            dataset[x][4] = temp[tempRand]              

        
def export():
    with open("..\custom-dataset.csv", "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dataset)


def main(): 
    makeDataSet()
    export()

if __name__ == '__main__': 
    main()