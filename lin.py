# Лінійна регресія
from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f"15-й варіант")

# print("Вводимо дані")
print(f"Подамо x як двовимірний масив:")
x = np.array([7.4, 8.9, 13.5, 15.2, 17.2, 19.6, 20.3, 23.5, 25.6, 26.3]).reshape((-1, 1))
y = np.array([6.5, 9.5, 11.8, 12.3, 15.5, 17.8, 18.9, 21.7, 22.3, 24.8])


# print("Створення самої моделі:")
model = LinearRegression()
print(model.fit(x, y))


# print("Отримаємо результати:")
# R² = (score)
# round(), 4 округлення
r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")

print(f"Подамо і y як двовимірний масив:")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept (a, 𝑏₀): {new_model.intercept_}")
print(f"slope (b, 𝑏₁): {new_model.coef_}")


# print("Прогнозуємо відповідь")
# g(xi) задяки функції Python
y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

# g(xi) по формулі
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

# модель для нових даних
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.YELLOW + f"Висновки:"
                 f"\n (𝑅²): {r_sq},"
                 f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏₁): {model.coef_}")