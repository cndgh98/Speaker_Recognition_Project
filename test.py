from functools import reduce
from jiwer import cer
a = [1, 2, 3, 4, 5]
l = 15

b = a[:]
for i in range(l - len(b)):
    i = i % len(b)
    b.append(b[i])

print(a)
print(b)

print(len(b))

dataset = [["안녕하세요"], ["안녕하세요"], ["아녕하세요"]]


def password_cv(dataset):
    dataset_len = len(dataset)
    error = [0 for _ in range(dataset_len)]
    for i in range(dataset_len):
        for j in range(dataset_len):
            if i == j:
                continue
            error[i] += cer(dataset[i][0], dataset[j][0])

    error = list(map(lambda x: x / (dataset_len - 1), error))

    min_error_idx = 0
    for i in range(dataset_len):
        if error[i] < error[min_error_idx]:
            min_error_idx = i

    return min_error_idx, error[min_error_idx]


print(password_cv(dataset))
a = [[1], [2], [3]]

print(reduce(lambda acc, cur: acc + cur, a, []))
