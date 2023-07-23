import numpy as np
import matplotlib.pyplot as plt

# Функция для проверки правильности вывода исходных уравнений (1)
def original_equations(x, y, a, b, K, c, d):
    dxdt = a * (1 - x/K) * x - (b * x * y) / (1 + a * x)
    dydt = -c * y + (d * x * y) / (1 + a * x)
    return dxdt, dydt

# Функция для проверки правильности вывода уравнений в безразмерном виде (2)
def dimensionless_equations(X, Y, epsilon, alpha, gamma):
    dXdt = (1 - epsilon * X) * X - (X * Y) / (1 + alpha * X)
    dYdt = -gamma * (1 - X / (1 + alpha * X)) * Y
    return dXdt, dYdt

# Функция для проверки правильности тестового решения (5)
def test_solution(x, y):
    dy1dt = -2 * x * y[0] * np.log(y[1])
    dy2dt = 2 * x * y[1] * np.log(y[0])
    return dy1dt, dy2dt

# Параметры для проверки исходных уравнений
a = 1.0
b = 0.1
K = 10.0
c = 1.5
d = 0.075
x = 1.0
y = 2.0

# Параметры для проверки безразмерных уравнений
epsilon = c / (K * d)
alpha = a * c / d
gamma = c / a

# Параметры для тестового решения
x_test = 1.0
y_test = np.array([2.0, 3.0])

# Проверка исходных уравнений
dxdt, dydt = original_equations(x, y, a, b, K, c, d)
print("Исходные уравнения:")
print("dx/dt =", dxdt)
print("dy/dt =", dydt)
print()

# Проверка безразмерных уравнений
dXdt, dYdt = dimensionless_equations(x, y, epsilon, alpha, gamma)
print("Безразмерные уравнения:")
print("dX/dτ =", dXdt)
print("dY/dτ =", dYdt)
print()

# Проверка тестового решения
dy1dt, dy2dt = test_solution(x_test, y_test)
print("Тестовое решение:")
print("dy1/dx =", dy1dt)
print("dy2/dx =", dy2dt)

