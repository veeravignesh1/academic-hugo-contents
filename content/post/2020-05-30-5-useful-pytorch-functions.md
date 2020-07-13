---
title: 5 Useful Pytorch Functions
author: Veera Vignesh
date: '2020-05-30'
slug: 5-useful-pytorch-functions
categories:
  - Deep Learning
tags:
  - Pytorch
subtitle: 'Exploring Pytorch Documentation'
summary: 'Exploring Pytorch Documentation'
authors: []
lastmod: '2020-05-30T18:52:48+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

In this post we are going to look into the following interesting functions which are readily available in pytorch for faster prototyping and development of the Deep Learning Project.

- `torch.addcmul()`
- `torch.gt()`
- `torch.flatten()`
- `torch.clamp()`
- `torch.cat()`


```python
# Import torch and other required modules
import torch
```

## Function 1 - torch.addcmul()

`torch.addcmul` function helps us in element wise multiplication of the passed in tensors (tensor 1 and tensor 2) take a fraction of it by multiplying with a constant value and summing up the resultant value with the input tensor.

The function can be represented as a formula as follows.

$\text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i$

Function takes in input, value, tensor1 and tensor2 as inputs, creating a tensor output ($out_i$).

Lets look at the examples

### Example 1


```python
# Input Tensor
input_t = torch.eye(1,2)

#Value
value=0.4

#Tensor 1 & 2
tensor1 = torch.eye(1,2)
tensor2 = torch.eye(2,1)

# Function
torch.addcmul(input_t,value,tensor1,tensor2)
```




    tensor([[1.4000, 0.0000],
            [1.0000, 0.0000]])



In the above example we use identity matrix to understand the functionality of the `torch.addmul()` function.It should note that the `tensor1` and `tensor2` is multiplied element wise before being multiplied with the user inputted value. Shape of `tensor1` and `tensor2` should have size which can be either broadcasted to match either of those are will result in an error. 

### Example 2


```python
# Example 2 - working
input_t = torch.randn(1,4)

#Value
value=0.4

#Tensor 1 & 2
tensor1 = torch.randn(1,4)
tensor2 = torch.randn(2,1)

# Function
torch.addcmul(input_t,value,tensor1,tensor2)
```




    tensor([[ 1.0210,  0.7045, -0.0551,  0.3437],
            [ 0.8734,  1.4989, -0.6034,  0.5361]])



**Step By Step Breakdown of the above Example**

1. `tensor1` and `tensor2` are multiplied element wise
2. Value provided by the user is then multiplied element wise with the matrix resulting from above.
- Input Tensor is then Added to the result produced


```python
#Checking shape of tensor1 and tensor2
print(f'Shape of Tensor1:{tensor1.shape}, Shape of Tensor2:{tensor2.shape}')

#Perfoming Element wise Matrix Multiplication
print("\nStep 1")
print(f'\n{torch.mul(tensor1,tensor2)}')

# Multiplying the Result produced with the inputted value
print("\nStep 2")
print(f'\n{value*torch.mul(tensor1,tensor2)}')

# Adding the Result with the input tensor
print("\nStep 3")
print(f'\n{input_t+value*torch.mul(tensor1,tensor2)}')
```

    Shape of Tensor1:torch.Size([1, 4]), Shape of Tensor2:torch.Size([2, 1])
    
    Step 1
    
    tensor([[ 0.1626, -0.8749,  0.6039, -0.2118],
            [-0.2065,  1.1111, -0.7669,  0.2690]])
    
    Step 2
    
    tensor([[ 0.0650, -0.3500,  0.2416, -0.0847],
            [-0.0826,  0.4444, -0.3068,  0.1076]])
    
    Step 3
    
    tensor([[ 1.0210,  0.7045, -0.0551,  0.3437],
            [ 0.8734,  1.4989, -0.6034,  0.5361]])


We can see that the result produced is consistent with the step by step operation performed by us

### Example 3


```python
# Example 3 - Breaking
input_t = torch.randn(1,3)

#Value
value=0.4

#Tensor 1 & 2
tensor1 = torch.randn(1,4)
tensor2 = torch.randn(3,1)

# Function
torch.addcmul(input_t,value,tensor1,tensor2)
```


    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-5-79eb12c122a1> in <module>
         10 
         11 # Function
    ---> 12 torch.addcmul(input_t,value,tensor1,tensor2)


    RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1


This function can break in two ways.
- If the `tensor1` and `tensor2` element wise multiplication does not match in dimension.
- The addition of `input_t` with the result produced doesn't match in dimension.

In the Above case it failed because its not able to add [1,3] tensor with [3,4] produced by the multiplication is not possible.

## Function 2 - torch.gt()

`torch.gt` function allows us in element wise comparision of the two inputs passed. The second argument can be a number that can be broadcasted to match the dimension of the first argument.

Mathematically represented as $\text{input} > \text{other}$

It takes in two arguments `input` and `other`
- input is the tensor on which element wise comparison is need to be done.
- other is the tensor value based on which the result is returned.

### Example 1


```python
# Example 1 - Comparing with a scalar

#Input
input_t = torch.arange(0,10)

# Function
torch.gt(input_t,torch.tensor([4]))
```




    tensor([False, False, False, False, False,  True,  True,  True,  True,  True])



In the above example the tensor input is compared with a single value 4 and the result is returned element wise.

### Example 2


```python
# Example 2 - working

#Input
input_t = torch.arange(0,9).reshape(3,3)
other = torch.tensor([5,2,1,4,7,2,3,1,6]).reshape(3,3)
# Function
torch.gt(input_t,other)
```




    tensor([[False, False,  True],
            [False, False,  True],
            [ True,  True,  True]])



In the above example we have two tensors of shape 3,3 and each element of `input_t` is compared with `other` tensor and the result is produced.

### Example 3


```python
# Example 3 - breaking (to illustrate when it breaks)

#Input
input_t = torch.arange(0,9).reshape(3,3)
other = torch.tensor([5,2,1,4,7,2,3,1,3,5]).reshape(2,5)
# Function
torch.gt(input_t,other)
```


    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-34-b0ecfa42a5af> in <module>
          7 other = torch.tensor([5,2,1,4,7,2,3,1,3,5]).reshape(2,5)
          8 # Function
    ----> 9 torch.gt(input_t,other)


    RuntimeError: The size of tensor a (3) must match the size of tensor b (5) at non-singleton dimension 1


Shape of the `input` and `other` should match.

This function can be used to subset based on the condition of the other tensor

## Function 3 - torch.flatten()

As the name suggest the function is used to flatten or convert into a single row from different shape of the tensor. This functionality can also be altered by assigning the start and end dimension of the tensor to flatten it.

`torch.flatten()` takes in one argument and 2 optional argument
- `input` is the tensor which is to be flattened
- `start_dim`is the dimension along which flattening should be done
- `end_dim` is the ending dimension along which flattening should be done.

### Example 1


```python
# Example 1 - working
# Without any Keyword Arguments

input_t = torch.arange(10).reshape(2,5)
print("Displaying Input tensor")
display(input_t)
print(f'{input_t.shape} \n')

# Flattening the tensor
print("Displaying Flattened Tensor")

torch.flatten(input_t)
```

    Displaying Input tensor



    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])


    torch.Size([2, 5]) 
    
    Displaying Flattened Tensor





    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



In this example the input tensor is of shape [2,5] which is then converted into [1,10] through flattening

### Example 2


```python
# Example 2 - working
# With Keyword Argument start_dim

input_t = torch.arange(20).reshape(2,2,5)
print("Displaying Input tensor")
display(input_t)
print(f'{input_t.shape} \n')

# Flattening the tensor
print("Displaying Flattened Tensor")

torch.flatten(input_t,start_dim=1)
```

    Displaying Input tensor



    tensor([[[ 0,  1,  2,  3,  4],
             [ 5,  6,  7,  8,  9]],
    
            [[10, 11, 12, 13, 14],
             [15, 16, 17, 18, 19]]])


    torch.Size([2, 2, 5]) 
    
    Displaying Flattened Tensor





    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])



In the above example input tensor is of shape [2,2,5] and since the tensor is flattened on shape 1 it returns a tensor of shape [2,10]

### Example 3


```python
# Example 3 - breaking (to illustrate when it breaks)

input_t = torch.arange(10).reshape(2,5)
print("Displaying Input tensor")
display(input_t)
print(f'{input_t.shape} \n')

# Flattening the tensor
print("Displaying Flattened Tensor")

torch.flatten(input_t,start_dim=2)
```

    Displaying Input tensor



    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])


    torch.Size([2, 5]) 
    
    Displaying Flattened Tensor



    ---------------------------------------------------------------------------
    
    IndexError                                Traceback (most recent call last)
    
    <ipython-input-52-202e52ce3fbf> in <module>
          9 print("Displaying Flattened Tensor")
         10 
    ---> 11 torch.flatten(input_t,start_dim=2)


    IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)


The above example breaks because the input tensor is of shape [2,5] and the start dimension is specified as 2. But the value in this case should be between [-2,1].

This function can be used when we need to convert the image tensor into a linear unit.

## Function 4 - torch.clamp()

`torch.clamp` function is used to limit the value in the tensor to a needed limit, within the min and max value.

Mathematically this function can be represented as 
$$
y_i = \begin{cases}
        \text{min} & \text{if } x_i < \text{min} \\
        x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
        \text{max} & \text{if } x_i > \text{max}
    \end{cases}
$$

`torch.clamp` takes in 3 arguments
- `input` which is the tensor on which the clamp is to be applied
- `min` the min value of the tensor after transformation
- `max` the max value of the tensor after transformation

It should be noted the values already lying within the range is untouched.

### Example 1


```python
# Example 1 - working
input_t = torch.randn(20)
display(input_t)

torch.clamp(input_t,-0.2,0.2)
```


    tensor([ 0.1495,  2.0461,  0.2567, -2.5744, -0.2684,  0.3525, -0.9201, -0.0600,
             0.3171,  0.7202, -0.2984,  0.2019,  0.5439,  0.1846,  0.4112, -0.4427,
             0.3650,  1.4413,  1.9937, -0.1986])





    tensor([ 0.1495,  0.2000,  0.2000, -0.2000, -0.2000,  0.2000, -0.2000, -0.0600,
             0.2000,  0.2000, -0.2000,  0.2000,  0.2000,  0.1846,  0.2000, -0.2000,
             0.2000,  0.2000,  0.2000, -0.1986])



In the above example the values are clamped to the min value of -0.2 and max of 0.2. the values that lie within the range are untouched.

### Example 2


```python
# Example 2 - working
input_t = torch.randn(20).reshape(2,2,5)
print("Before Clamping")
display(input_t)

print("After Clamping")
display(torch.clamp(input_t,max=0.3))
```

    Before Clamping



    tensor([[[ 0.6896, -0.7686,  0.0441,  1.0526, -0.1661],
             [ 0.1653, -1.4807,  0.3776, -0.8891,  0.5351]],
    
            [[-0.0792,  0.7499,  1.4385, -0.7969,  1.8040],
             [-0.7114, -0.3415, -0.2100,  1.3812, -0.6974]]])


    After Clamping



    tensor([[[ 0.3000, -0.7686,  0.0441,  0.3000, -0.1661],
             [ 0.1653, -1.4807,  0.3000, -0.8891,  0.3000]],
    
            [[-0.0792,  0.3000,  0.3000, -0.7969,  0.3000],
             [-0.7114, -0.3415, -0.2100,  0.3000, -0.6974]]])


In this example only the max value is clamped and the lower-bound is untouched.

### Example 3 


```python
# Example 3 - breaking (to illustrate when it breaks)
input_t = torch.randn(20).reshape(2,2,5)
print("Before Clamping")
display(input_t)

print("After Clamping")
display(torch.clamp(input_t))

```

    Before Clamping



    tensor([[[ 1.3157, -0.2641, -0.8378,  0.8199,  0.2945],
             [-0.2855, -1.1077,  0.0079,  1.1202,  0.4865]],
    
            [[ 0.3016,  0.0990, -1.9524,  1.3870, -0.4927],
             [-0.7312, -0.6831,  1.1755,  0.4609,  0.4770]]])


    After Clamping



    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-60-983e123d0323> in <module>
          5 
          6 print("After Clamping")
    ----> 7 display(torch.clamp(input_t))


    RuntimeError: At least one of 'min' or 'max' must not be None


This function breaks without max or min being passed to the argument.

Can be used in optimizers to clip the value of the produced tensors.

## Function 5 - torch.cat()

`torch.cat` concatenates the sequence of tensors in the given dimension.All tensors must have the same shape or be empty.

`torch.cat` takes in 2 arguments
- `tensors` - The input tensor sequence which can be 2 or more tensors
- `dim` - The dimension along which the tensor should be concatenated.

### Example 1


```python
# Example 1 - working
tensor1 = torch.arange(1,11).reshape(2,5)
tensor2 = torch.arange(11,21).reshape(2,5)

torch.cat((tensor1,tensor2), dim=1)
```




    tensor([[ 1,  2,  3,  4,  5, 11, 12, 13, 14, 15],
            [ 6,  7,  8,  9, 10, 16, 17, 18, 19, 20]])



In the above case we have two tensor with shape 2,5 and trying to concatenate along the row. So this just horizontally stacks the tensor over the tensor1.

### Example 2


```python
# Example 2 - working

tensor1 = torch.arange(1,21).reshape(2,2,5)
tensor2 = torch.arange(21,41).reshape(2,2,5)

torch.cat((tensor1,tensor2), dim=1)
```




    tensor([[[ 1,  2,  3,  4,  5],
             [ 6,  7,  8,  9, 10],
             [21, 22, 23, 24, 25],
             [26, 27, 28, 29, 30]],
    
            [[11, 12, 13, 14, 15],
             [16, 17, 18, 19, 20],
             [31, 32, 33, 34, 35],
             [36, 37, 38, 39, 40]]])



In the above 3 dimensional tensor stacking of the values have happened along the row. each row value is exapaned to tensor.


```python
# Example 3 - breaking (to illustrate when it breaks)

tensor1 = torch.arange(1,21).reshape(2,2,5)
tensor2 = torch.arange(21,41).reshape(2,2,5)

torch.cat((tensor1,tensor2), dim=3)
```


    ---------------------------------------------------------------------------
    
    IndexError                                Traceback (most recent call last)
    
    <ipython-input-78-a55df25647fd> in <module>
          4 tensor2 = torch.arange(21,41).reshape(2,2,5)
          5 
    ----> 6 torch.cat((tensor1,tensor2), dim=3)


    IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)


Dimension did not match in this case. So the function fails. should be within the range of [-3,2]

## Conclusion

So In this article we have seen some exciting function which can ease out the speed of experimenting during model development. This is by no means the exhaustive list of functions available. For more details check out official documentation.

## Reference Links
Provide links to your references and other interesting articles about tensors
* Official documentation for `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html


```python
!pip install jovian --upgrade --quiet
```


```python
import jovian
```


    <IPython.core.display.Javascript object>



```python
jovian.commit()
```


    <IPython.core.display.Javascript object>


    [jovian] Attempting to save notebook..
    [jovian] Please enter your API key ( from https://jovian.ml/ ):
    API KEY: ········
    [jovian] Uploading notebook..
    [jovian] Capturing environment..
    [jovian] Committed successfully! https://jovian.ml/veeravignesh1/01-tensor-operations-c9926





    'https://jovian.ml/veeravignesh1/01-tensor-operations-c9926'




```python

```
