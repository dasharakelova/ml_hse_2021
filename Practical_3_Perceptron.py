#!/usr/bin/env python
# coding: utf-8

# Task 1

# In[1]:


from random import randint
from random import random
from typing import Union, List
from math import sqrt
import matplotlib.pyplot as plt


# In[2]:


class Scalar:
  pass
class Vector:
  pass

class TypeError(Exception):
    def __init__(self, message):
        self.message = message

class Scalar:
    def __init__(self: Scalar, val: float):
        self.val = float(val)

    def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]: 
        if isinstance(other, Scalar):
            return Scalar(self.val * other.val)
        elif isinstance(other, Vector):
            return Vector(*[self.val * entry for entry in other])
        else:
            raise TypeError(f"{other} isn't Scalar or Vector")

    def __add__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val + other.val)

    def __sub__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val - other.val)

    def __truediv__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val / other.val)

    def __rtruediv__(self: Scalar, other: Vector) -> Vector:
        return Vector(*[entry / self.val for entry in other])

    def __repr__(self: Scalar) -> str:
        return "Scalar(%r)" % self.val

    def sign(self: Scalar) -> int:
        if self.val < 1:
            return -1
        elif self.val > 1:
            return 1
        return 0

    def __float__(self: Scalar) -> float:
        return self.val

class Vector:
    def __init__(self: Vector, *entries: List[float]):
        self.entries = entries

    def zero(size: int) -> Vector:
        return Vector(*[0 for i in range(size)])

    def __add__(self: Vector, other: Vector) -> Vector:
        return Vector(*[one_x + another_x 
                        for one_x, another_x in zip(self, other)])
    
    def __sub__(self: Vector, other: Vector) -> Vector:
        return Vector(*[one_x - another_x 
                        for one_x, another_x in zip(self, other)])

    def __mul__(self: Vector, other: Vector) -> Scalar:
        return Scalar(sum(one_entry * another_entry 
                          for one_entry, another_entry in zip(self, other)))

    def magnitude(self: Vector) -> Scalar:
        return Scalar(sqrt(sum(entry**2 for entry in self)))

    def unit(self: Vector) -> Vector:
        return self / self.magnitude()

    def __len__(self: Vector) -> int:
        return len(self.entries)

    def __repr__(self: Vector) -> str:
        return "Vector%s" % repr(self.entries)

    def __iter__(self: Vector):
        return iter(self.entries)


# Task 2

# In[3]:


def PerceptronTrain(D, maxiter=100):
    w = Vector.zero(len(D[0][0]))
    b = Scalar(0)
    for i in range(maxiter):
        for x,y in D:
            a = Scalar(w.__mul__(x) + b)
            if y.__mul__(a).sign() <= 0:
                w = w.__add__(y.__mul__(x))
                b = b.__add__(y)
    return w,b


# In[4]:


def PerceptronTest(w,b,x):
    a = Scalar(0)
    for ent in x:
        a = a.__add__(Scalar(w.__mul__(x) + b))
    return a.sign()


# D is a list of lists (or tuples) of length 2, where the first element is a Vector (i.e. the data) and the second a Scalar (i.e. the label).
# We use Vector addition and multiplication and Scalar addition in the algorithm, as well as sign and zero methods.
# 
# Task 3

# In[5]:


v = Vector(randint(-100, 100), randint(-100, 100))
xs1 = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys1 = [v * x * Scalar(randint(-1, 9)) for x in xs1]
data1 = [(x, y) for x, y in zip(xs1, ys1)]


# Let's make a train-test split:

# In[6]:


train1 = data1[:450]
test1 = data1[450:]


# And write a function to compute the score:

# In[7]:


def PercepEval(train, test):
    weights, bias = PerceptronTrain(train)
    results = [PerceptronTest(weights, bias, x[0]) for x in test]
    return results.count(1)/len(results)


# In[8]:


PercepEval(train1, test1)


# Not so good, actually:(
# 
# Task 4

# In[9]:


xs2 = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys2 = [Scalar(1) if x.entries[0]*x.entries[1] < 0 else Scalar(-1) for x in xs2]
data2 = [(x, y) for x, y in zip(xs2, ys2)]


# In[10]:


train2 = data2[:450]
test2 = data2[450:]


# In[11]:


PercepEval(train2, test2)


# Not much lower.

# Task 5
# 
# First let's rewrite the train function so that it saves the weights and bias after every epoch:

# In[12]:


def PerceptronTrainEp(D, maxiter=100):
    w = Vector.zero(len(D[0][0]))
    b = Scalar(0)
    hist = []
    for i in range(maxiter):
        for x,y in D:
            a = Scalar(w.__mul__(x) + b)
            if y.__mul__(a).sign() <= 0:
                w = w.__add__(y.__mul__(x))
                b = b.__add__(y)
        hist.append((w,b))
    return w,b, hist


# Now the plotting. Non-permutted dataset:

# In[13]:


train3 = sorted(train1, key=lambda x:x[1].val)


# In[14]:


w, b, hist = PerceptronTrainEp(train3)
scores1 = []
for weights, bias in hist:
    results = [PerceptronTest(weights, bias, x[0]) for x in test1]
    scores1.append(results.count(1)/len(results))


# In[15]:


plt.plot([i for i in range(100)], scores1)
plt.xlabel("epochs")
plt.ylabel("score")
plt.title('No permutations in the sorted training dataset')
plt.show()


# Random permutation at the beginning:

# In[16]:


train4 = sorted(train1, key=lambda x: random())


# In[17]:


w, b, hist = PerceptronTrainEp(train4)
scores2 = []
for weights, bias in hist:
    results = [PerceptronTest(weights, bias, x[0]) for x in test1]
    scores2.append(results.count(1)/len(results))


# In[18]:


plt.plot([i for i in range(100)], scores2)
plt.xlabel("epochs")
plt.ylabel("score")
plt.title('Random permutation at the beginning')
plt.show()


# Random permutation at each epoch:

# Let's rewrite the train algorithm again.

# In[19]:


def PerceptronTrainPermute(D, maxiter=100):
    w = Vector.zero(len(D[0][0]))
    b = Scalar(0)
    hist = []
    for i in range(maxiter):
        D = sorted(D, key=lambda x: random())
        for x,y in D:
            a = Scalar(w.__mul__(x) + b)
            if y.__mul__(a).sign() <= 0:
                w = w.__add__(y.__mul__(x))
                b = b.__add__(y)
        hist.append((w,b))
    return w,b, hist


# In[20]:


w, b, hist = PerceptronTrainPermute(train3)
scores3 = []
for weights, bias in hist:
    results = [PerceptronTest(weights, bias, x[0]) for x in test1]
    scores3.append(results.count(1)/len(results))


# In[21]:


plt.plot([i for i in range(100)], scores3)
plt.xlabel("epochs")
plt.ylabel("score")
plt.title('Permutations at each epoch')
plt.show()


# I guess something went wrong in the algorithm because the last performance clearly isn't the best. (also all the plots look chaotic)
# 
# Task 6
# 
# The averaged algorithm:

# In[41]:


def AveragedPerceptronTrainPermute(D, maxiter=100):
    w = Vector.zero(len(D[0][0]))
    b = Scalar(0)
    W = Vector.zero(len(D[0][0])) #cached weights
    B = Scalar(0) #cached bias
    c = Scalar(1)
    for i in range(maxiter):
        D = sorted(D, key=lambda x: random())
        for x,y in D:
            a = Scalar(w.__mul__(x) + b)
            if y.__mul__(a).sign() <= 0:
                w = w.__add__(y.__mul__(x))
                b = b.__add__(y)
                W = W.__add__(y.__mul__(c.__mul__(x)))
                B = B.__add__(y.__mul__(c))
            c = c.__add__(Scalar(1))    
    w_av = w.__sub__(Scalar(1).__truediv__(c).__mul__(W))
    b_av = b.__sub__(B.__mul__(Scalar(1).__truediv__(c)))
    return w_av, b_av


# Comparing the results:

# In[42]:


weights, bias = AveragedPerceptronTrainPermute(train1)
results = [PerceptronTest(weights, bias, x[0]) for x in test1]
results.count(1)/len(results)


# It's better then the score of 0.54 we got in task 3, but not by much.
