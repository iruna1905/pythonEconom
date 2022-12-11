# Множинна регресія
from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f"15-й варіант")

# print("Вводимо дані")
y = [8.08, 8.48, 8.32, 8.80, 9.44, 9.68, 9.78, 11.28, 10.24,
     10.72, 12.00, 11.76, 12.64, 13.36, 15.52, 17.68, 18.48, 18.96]

x = [
    [2.95, 27.44], [3.35, 29.08], [3.80, 32.02], [4.15, 29.06], [3.90, 34.52],
    [4.60, 37.08], [4.30, 34.70], [4.60, 35.62], [4.50, 43.42], [4.50, 43.66],
    [5.15, 41.36], [4.75, 49.12], [5.25, 44.90], [4.90, 47.10], [5.00, 47.84],
    [5.20, 48.65], [5.26, 48.25], [5.85, 50.32]
]
x, y = np.array(x), np.array(y)


# print("Створення самої моделі:")
model = LinearRegression().fit(x, y)
print(model)


# print("Отримаємо результати:")
# R² = (score)
# round(), 4 округлення
r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")


# print("Прогнозуємо відповідь")
# g(xi) задяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) по формулі
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

# модель для нових даних
x_new = np.arange(10).reshape((-1, 2))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.YELLOW + f"Висновки:"
                 f"\n (𝑅²): {r_sq},"
                 f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏₁): {model.coef_}")
