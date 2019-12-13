---
title: Introduction to Computer Science and Programming in Python
author: Veera Vignesh
date: '2019-12-13'
slug: introduction-to-computer-science-and-programming-in-python
categories:
  - Programming
tags:
  - Python
subtitle: 'Taught by Ana Bell [YouTube](https://www.youtube.com/watch?v=nykOeWgQcHM&list=PLUl4u3cNGP63WbdFxL8giv4yhgdMGaZNA)'
summary: 'Course notes of [6.001](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) - MIT OCW'
authors: []
lastmod: '2019-12-13T05:22:54+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

## What is computation

**What computers do** 

- Performs calculations and remembers the results
- User defined calculations and built in calculations
- Computers only know what you tell them

**Types of knowledge**

- Declarative - Statement of facts - humans understand facts
- Imperative - Recipe or how to - computers do not understand facts, they take in a sequence of steps as input for their calculations

<u>Ex</u> Square root of a number 

**Algorithm** 

- Sequence of simple steps
- Flow of controls
- Number of times to run the program - Iterations
- Stopping condition

In the early days the computers where only **Fixed program computers** meaning designed to do only one task. Then came the **Stored program computers** which where then used to store and execute the instructions.

**Basic Machine Architecture**

![Arch](arch.png)

- Memory contains the sequence of instructions and data
- ALU gets the data and performs some primitive operations and store it back to the memory
- Control unit keeps tracks of the flow of program execution
- After the program is done necessary output is displayed

**Primitive Operations**

- Arithmetic and logic
- Simple test
- Move data

> Alan turning showed that with add, subtract, move left , move right, scan, do nothing any basic construct can be made.

Programming Languages then came along which provided expressions which had primitives built in. primitives in python `float`,`bool`, `int`

In programming language the set of instructions which is semantically and syntactically correct will imply only one meaning

**Types of errors**

- Syntax error
- Semantic error
- Output is different then the expected 

In python everything is an object. Each object has its own type. Based on the type of the object we can decide on what type of operations can be performed on it.

**Types of Objects** 

- Scalar - Objects which can't be further subdivided (`int`,`bool`,`float`,`None`)
- Non-Scalar - Objects which can be further subdivided (`list`,`dict`)

**Expressions**

- Combine objects and operators to form expressions
- expression has a value that has a type

**Operations**

The following or the operations that can be performed in python `+` `-` `*` `/` `%` `**` . Order of precedence of the operation is followed we can overwrite with parenthesis.

**Variables**

It is useful to assign values to the variables and store them so that we can access them for later use. `=` sign is used to assign the value to the name that we pick. variable name should be descriptive. variable acts as a pointer to that memory in space where the value is stored.

If the value of the variable is updated it now points to the new value (changing bindings). The old value is left hanging and eventually is then collected by garbage collectors build into the program.









