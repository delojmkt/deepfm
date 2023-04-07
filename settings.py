import os 

path = "/Users/gimda-eun/Downloads/laptops.csv"

sparse_features = ['name','processor','ram','os','storage']
dense_features = ['price(in Rs.)', 'display(in inch)']
                  
input_col = ['name','price(in Rs.)','processor','ram','os','storage','display(in inch)']
target_col = ['rating']

embedding_dim = 4

test_size = 0.2