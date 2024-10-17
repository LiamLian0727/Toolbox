## Debug Utilities for Tensor Data in PyTorch

This repository contains a set of utility functions designed to simplify the debugging process when working with PyTorch tensors and nested data structures. The utilities provide detailed and well-formatted information about the data, which is helpful during model development and troubleshooting.

### Overview

- **`debug_print(*args, **kwargs)`**: A simple function to print debugging information only when debugging mode is enabled. By default, debugging mode can be controlled with the environment variable `TGS_DEBUG`. If `TGS_DEBUG` is set to `'True'` (case insensitive), debug messages are printed; otherwise, they are suppressed.

- **`debug_print_tensor_dict(data, str="", indent=2)`**: This function takes a dictionary containing PyTorch tensors and prints out the shapes, types, and devices of the tensors in a well-formatted manner. It also handles nested dictionaries and lists, allowing you to easily visualize complex data structures.

- **`debug_print_tensor_list(data, str="", indent=2)`**: Similar to `debug_print_tensor_dict`, this function works with lists that contain PyTorch tensors, dictionaries, or other lists. It will recursively format and display the details of the tensors, making it easy to understand nested structures.

### How to Use

1. Set the environment variable `TGS_DEBUG` to `'True'` to enable debug mode. This can be done by running:
   ```sh
   export TGS_DEBUG=True
   ```
   or by modifying it in the code as follows:
   ```python
   os.environ['TGS_DEBUG'] = 'True'
   ```

2. Use the `debug_print()`, `debug_print_tensor_dict()`, or `debug_print_tensor_list()` functions in your code to print out detailed information about tensors and other variables.

### Example Usage

```python
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
                "b": [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)]
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
            "son_dict": {
                "a": torch.randn(3, 4),
                "b": [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)]
            }
        },
        42
    ]

    debug_print_tensor_dict(data, "dict test")
    debug_print_tensor_list(data2, "list test")
```

### Features

- **Conditional Debug Printing**: Only outputs debug information when explicitly enabled.
- **Rich Tensor Information**: Provides details such as tensor shape, data type, and device, which are often useful during debugging.
- **Handles Nested Structures**: Recursively prints dictionaries and lists, which can contain other data structures or tensors.

### License
This code is open-sourced and can be used or modified freely for personal and commercial projects.

