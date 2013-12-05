#!/bin/bash

for s in $(cat ../stocks)
do
	#wget http://ichart.finance.yahoo.com/table.csv?s=$s&a=01&b=19&c=2010&d=01&e=21&f=2010&g=d&ignore=.csv
	wget http://ichart.finance.yahoo.com/table.csv?s=$s&a=01&b=19&c=2010&d=01&e=21&f=2010&g=d
done
