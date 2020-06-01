#!/bin/bash

shuf -n 500000 ${1} > "test.jl"


#split -l 100000 ${1} ${2}