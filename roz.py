# Розширина регресія
from colorama import Fore
import numpy as np
import statsmodels.api as sm

print(f"15-й варіант")

# print(f"даємо та трансформуємо дані")
y = [8.08, 8.48, 8.32, 8.80, 9.44, 9.68, 9.78, 11.28, 10.24,
     10.72, 12.00, 11.76, 12.64, 13.36, 15.52, 17.68, 18.48, 18.96]

x = [
    [2.95, 27.44], [3.35, 29.08], [3.80, 32.02], [4.15, 29.06], [3.90, 34.52],
    [4.60, 37.08], [4.30, 34.70], [4.60, 35.62], [4.50, 43.42], [4.50, 43.66],
    [5.15, 41.36], [4.75, 49.12], [5.25, 44.90], [4.90, 47.10], [5.00, 47.84],
    [5.20, 48.65], [5.26, 48.25], [5.85, 50.32]
]
x, y = np.array(x), np.array(y)

x = sm.add_constant(x)
print(f"X = \n{x}")
print(f"Y = \n{y}")

# print("Створення самої моделі:")
model = sm.OLS(y, x)
results = model.fit()

# Regression
print(Fore.YELLOW, f"Regression (Регресія) як в Exel:"
                   f"\n{results.summary()}")