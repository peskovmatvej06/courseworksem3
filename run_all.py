import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=== ЗАПУСК ПОЛНОГО ЦИКЛА ЭКСПЕРИМЕНТОВ ===")
    print(f"Рабочая директория: {os.getcwd()}")
    
    # 1. Запуск эксперимента с Квадратичной функцией
    try:
        print("\n--> Запуск quadratic.py...")
        import quadratic
        quadratic.exp_quadratic()
    except ImportError:
        print("ОШИБКА: Не найден файл quadratic.py или функция exp_quadratic внутри него.")
    except Exception as e:
        print(f"ОШИБКА при выполнении: {e}")

    # 2. Запуск эксперимента Розенброка
    try:
        print("\n--> Запуск rosenbrock.py...")
        import rosenbrock
        rosenbrock.exp_rosenbrock()
    except ImportError:
        print("ОШИБКА: Не найден файл rosenbrock.py или функция exp_rosenbrock внутри него.")
    except Exception as e:
        print(f"ОШИБКА при выполнении: {e}")

    # 3. Запуск эксперимента SGD
    try:
        print("\n--> Запуск sgd.py...")
        import sgd
        
        if hasattr(sgd, 'exp_sgd_cloud'):
            sgd.exp_sgd_cloud()
        else:
            
            print("Внимание: ищем альтернативную функцию запуска...")
            if hasattr(sgd, 'main'): sgd.main()
    except ImportError:
        print("ОШИБКА: Не найден файл sgd.py")
    except Exception as e:
        print(f"ОШИБКА при выполнении: {e}")
    
    print("\n=== ГОТОВО. Проверьте созданные PNG файлы. ===")

if __name__ == "__main__":
    main()
