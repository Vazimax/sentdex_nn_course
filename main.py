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

output = np.dot(inputs, np.array(weights).T) + biases
print(f'the output is {output}')