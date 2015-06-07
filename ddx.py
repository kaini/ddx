#!/usr/bin/env python3
import subprocess
import sys
import io
import urllib.request
import urllib.parse
from random import randint, choice

def inum(allow0=False):
    n = randint(-10, 10)
    if n == -10:
        return "(e)"
    elif n == 10:
        return "(pi)"
    elif not allow0 and n == 0:
        return num(False)
    else:
        return "n(" + str(n) + ")"

def num(allow0=False):
    return inum(allow0)

def simpleterm():
    type = randint(0, 1)
    if type == 0:
        # a*fn(b*x)
        return "(" + num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + num() + "*x))"
    elif type == 1:
        # a*x^b
        return "(" + num() + "*x^" + num() + ")"

def simpleterm_offset():
    type = randint(0, 1)
    if type == 0:
        # a*fn(b*x+c)
        return "(" + num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + num() + "*x+" + num(True) + "))"
    elif type == 1:
        # a*(x+c)^b
        return "(" + num() + "*(x+" + num(True) + ")^" + num() + ")"
    
def simple():
    type = randint(0, 2)
    print("simple", type)
    if type == 0:
        return "(" + simpleterm() + choice(["+", "-"]) + simpleterm() + ")"
    elif type == 1:
        return simpleterm_offset()
    elif type == 2:
        # a*fn(b*fn(c*x))
        return "(" + num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + \
            num() + "*" + choice(["ln", "cos", "sin", "e^"]) + "(" + \
            num() + "*" + "x)))"

def medium():
    type = randint(0, 2)
    print("medium", type)
    if type == 0:
        return "(" + simple() + choice(["*"]) + simple() + ")"
    elif type == 1:
        return "(" + simple() + "^" + num() + ")"
    elif type == 2:
        return "(" + choice(["tan", "cot"]) + "(" + simple() + "))"

def make_integral(term, output_path):
    with open("input.txt", "w") as fp:
        fp.write("""
            [ddx].
            main_print(""" + term + """), !.
            halt.
        """)

    with open("input.txt", "r") as infp:
        with open("output.txt", "w") as outfp:
            subprocess.call(["swipl", "--quiet", "ddx.pl"],
                            stdin=infp, stdout=outfp, stderr=outfp)

    with open("output.txt", "r") as fp:
        output = fp.read()
    latex = [l for l in output.split("\n") if l.startswith("\\int")][0]

    get_arg = urllib.parse.quote("\\dpi{150} \\bg_white " + latex, safe="")
    url = "http://latex.codecogs.com/png.latex?" + get_arg
    with urllib.request.urlopen(url) as http:
        png_data = http.read()

    with open(output_path, "wb") as fp:
        fp.write(png_data)

def generate(count, whatstr, what):
    for i in range(count):
        make_integral(what(), whatstr + str(i) + ".png")
