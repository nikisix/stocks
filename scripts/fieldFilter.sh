#!/bin/bash

for file in ../tables/*
do
	awk -F, '{print $1,$7}' ${file} > ${file}.filtered
done
