import threading
import time

count = 0

isConfirm = False


def countdown():
    global count
    global isConfirm
    while True:
        count += 1
        print("count =>", count)
        time.sleep(1)

        if (count == 5):
            isConfirm = True
            break


def print_end():
    print("End...")


if __name__ == "__main__":
    t1 = threading.Thread(target=countdown)
    t2 = threading.Thread(target=print_end)

    t1.start()
    t2.start()

    t1.join()
    t2.join()
