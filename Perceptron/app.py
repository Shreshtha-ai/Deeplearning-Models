from perceptron_scratch import Perceptron



trainig_data = [
    ([0,0,0],0),
    ([0,1,1],0),
    ([1,0,1],0),
    ([1,1,1],1)
]

p = Perceptron(3,0.1)
p.train(trainig_data)

print("weights", p.weights)
print("bias", p.bias)
print("predictions",p.predict([0,0,0]),p.predict([0,1,1]),p.predict([1,0,1]),p.predict([1,1,1]))