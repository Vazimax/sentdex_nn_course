import numpy as np

inputs = [[1,2,3,4],
          [7,4,-3,0.65],
          [8.7,0.26,3,-4],
]

weights = [
    [5,3.1,6.5,12]
    ,[0.1,7,0.5,2.4]
    ,[2,9,3,0.2]
]

biases = [3.1, 0.8, 2]

weights2 = [
    [1.2,1.1,9.5]
    ,[0.1,7,2.4]
    ,[3,7,1.2]
]

biases2 = [3.1, 0.8, 2]

output = np.dot(inputs, np.array(weights).T) + biases
output2 = np.dot(output, np.array(weights2).T) + biases2

print(f'the output is {output}')