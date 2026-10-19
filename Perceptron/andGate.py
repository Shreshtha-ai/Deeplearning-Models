and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

from perceptron_scratch import Perceptron

p = Perceptron(2, 0.1)
p.train(and_data)

print("weights", p.weights)
print("bias", p.bias)
print("predictions",p.predict([0,0]),p.predict([0,1]),p.predict([1,0]),p.predict([1,1]))
