a = int(input("Введите номер урока: "))
start_1 = 8
start_2 = 30
while a != 0:
    if a == 1:
        start_2 += 45
        start_1 += start_2 // 60
        start_2 = start_2 % 60
        a -= 1
    else:
        start_2 += 55
        start_1 += start_2 // 60
        start_2 = start_2 % 60
        a -= 1

    if start_1 // 24 != 0:
        start_1 = start_1 % 24
print(f"{start_1}-{start_2}")