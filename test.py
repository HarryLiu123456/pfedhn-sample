def test(func):
    def wrapper(*args, **kwargs):
        print('Something is happening before the function is called.')
        func(*args, **kwargs)
        print('Something is happening after the function is called.')
    return wrapper

@test
def say_hello(name):
    print(f'Hello {name}')


test(say_hello('John'))


