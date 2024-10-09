import inspect
import os

import torch

DEBUG = os.getenv('TGS_DEBUG', 'False').lower() == 'true'


def debug_print(*args, **kwargs):
    if not DEBUG:
        return  # Do not print anything if DEBUG is False

    # Get the current stack information
    stack = inspect.stack()
    caller_frame = stack[1]
    filename = caller_frame.filename
    lineno = caller_frame.lineno
    function_name = caller_frame.function

    #  Print the file name, function name, and line number
    print(f"[{filename}:{lineno}-{function_name}()]: ", *args, **kwargs)

def debug_print_tensor_dict(data, str="", indent=2):
    if not DEBUG:
        return  # Do not print anything if DEBUG is False

    indent_space = ' ' * indent

    if indent == 2:
        debug_print(str)
        print((' ' * (indent - 2)) + "{")

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            # Print the shape, type, and device of the Tensor
            print(f"{indent_space}{key}: Tensor(shape={value.shape}, type={value.dtype}, device={value.device}),")
        elif isinstance(value, list):
            # Iterate through the list and check if elements are Tensors
            print(f"{indent_space}{key}: [", )
            debug_print_tensor_list(value, indent=indent + 2)
            print(f"{indent_space}],")
        elif isinstance(value, dict):
            # Recursively print nested dictionaries
            print(f"{indent_space}{key}: ", "{")
            debug_print_tensor_dict(value, indent=indent + 2)
            print(f"{indent_space}", "},")
        else:
            # Print the value as-is if it's neither Tensor nor List
            print(f"{indent_space}{key}: {value},")

    if indent == 2:
        print((' ' * (indent - 2)) + "}")


def debug_print_tensor_list(data, str="", indent=2):
    if not DEBUG:
        return  # Do not print anything if DEBUG is False

    indent_space = ' ' * indent

    if indent == 2:
        debug_print(str)
        print((' ' * (indent - 2)) + "[")

    for key, value in enumerate(data):
        if isinstance(value, torch.Tensor):
            # Print the shape, type, and device of the Tensor
            print(f"{indent_space}[{key}]: Tensor(shape={value.shape}, type={value.dtype}, device={value.device}),")
        elif isinstance(value, list):
            # Iterate through the list and check if elements are Tensors
            print(f"{indent_space}[{key}]: [", )
            debug_print_tensor_list(value, indent=indent + 2)
            print(f"{indent_space}],")
        elif isinstance(value, dict):
            # Recursively print nested dictionaries
            print(f"{indent_space}[{key}]: ", "{")
            debug_print_tensor_dict(value, indent=indent + 2)
            print(f"{indent_space}", "},")
        else:
            # Print the value as-is if it's neither Tensor nor List
            print(f"{indent_space}[{key}]: {value},")

    if indent == 2:
        print((' ' * (indent - 2)) + "]")


# 示例使用
def example_function():
    debug_print("This is a debug message")


if __name__ == '__main__':
    os.environ['TGS_DEBUG'] = 'True'
    debug_print("This is a debug message")

    data = {
        "tensor_1": torch.randn(3, 4),
        "tensor_list": [torch.randn(2, 3), torch.randn(5), 3],
        "nested_dict": {
            "tensor_2": torch.randn(6),
            "value": 100,
            "son_dict": {
                "a": torch.randn(3, 4),
                "b": [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4), ]
            }
        },
        "other_value": 42
    }

    data2 = [
        torch.randn(3, 4),
        [torch.randn(2, 3), torch.randn(5), 3],
        {
            "tensor_2": torch.randn(6),
            "value": 100,
            "son_dict":
                {
                    "a": torch.randn(3, 4),
                    "b": [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)]
                }
        },
        42
    ]

    debug_print_tensor_dict(data, "dict test")
    debug_print_tensor_list(data2, "list test")
