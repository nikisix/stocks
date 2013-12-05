import scipy
a = [1,2,3]
b = scipy.array([i] for i in a)
for j in b.tolist():
    print(j)
