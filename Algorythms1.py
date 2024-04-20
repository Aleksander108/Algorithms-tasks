#!/usr/bin/env python
# coding: utf-8

# In[26]:


#Create a function that takes two or more arrays and returns an array of their symmetric difference. 
#The returned array must contain only unique values (no duplicates).

def sym(args):
    final_array=[]
    for i in range(len(args)):
        ee=args.pop(i)
        print(ee)
        for el in ee:
            for array in args:
                if el not in array and el not in final_array:
                    final_array.append(el)
        args.insert(i,ee)
    return (final_array)


args=[[1, 2, 3], [5, 2, 1, 4],[2,3,4,5,6]]
sym(args)





# In[25]:


args=[[1, 2, 3], [5, 2, 1, 4]]
args_set = {element for sublist in args for element in sublist}


# In[46]:


args = [[1, 2, 3], [5, 2, 1, 4]]
args_set = {element for sublist in args for element in sublist}
print(args_set)


# In[16]:


args = [[1, 2, 3], [5, 2, 1, 4]]

for current_sublist in args:
    # Create a flattened list of all other sublists
    other_elements = [item for sublist in args if sublist is not current_sublist for item in sublist]
    # Check if any element in the current sublist is in the flattened list of other sublists
    common_elements = set(current_sublist) & set(other_elements)
    if common_elements:
        print(f"Common elements found: {common_elements}")
    else:
        print("No common elements found.")


# In[17]:


args = [[1, 2, 3], [5, 2, 1, 4]]
flattened_set = {item for sublist in args for item in sublist}
print(flattened_set)


# In[1]:


args = [[1, 2, 3], [5, 2, 1, 4]]
final_array=[]
for array in args:
    args_cut=args.remove(array)
    print (args)

    found = any(i in array for array in args_cut)
    if not found:
        final_array+=i
    args_cut=  args  
print(final_array)


# In[64]:


args=[[1, 2, 3], [5, 2, 1, 4],[2,3,4,5,6]]
final_array=[]
for i in range(len(args)):
    ee=args.pop(i)
    for el in ee:
        for array in args:
            if el not in array :
                if el not in array and el not in final_array:
                    final_array.append(el)
    args.insert(i,ee)
print (final_array)


# In[27]:


my_list = [[1, 2, 3], [5, 2, 1, 4]]
sublist_to_remove = [1, 2, 3]  # The exact sublist you want to remove

my_list.remove(sublist_to_remove)
print(my_list)


# In[38]:


my_list = [[1, 2, 3], [5, 2, 1, 4]]
cat=my_list.remove(my_list[0])   # The exact sublist you want to remove

print(my_list)
my_listprint(my_list)
(cat)
print(my_list)


# In[43]:


ar = [[1, 2, 3], [5, 2, 1, 4]]
tuple_of_tuples = tuple(tuple(sublist) for sublist in ar)
print(tuple_of_tuples)


# In[60]:


args=[[1, 2, 3], [5, 2, 1, 4],[2,3,4,5,6]]
final_array = []

for i in range(len(args)):
    # Временно удаляем текущий подсписок из args
    temp_list = args.pop(0)
    
    # Проверяем каждый элемент в текущем подсписке
    for el in temp_list:
        if all(el not in array for array in args):
            # Если элемент уникален (не встречается в оставшихся подсписках), добавляем его в final_array
            final_array.append(el)

    # Возвращаем временно удаленный подсписок на его место в конец списка
    args.append(temp_list)

print(final_array)


# In[65]:


args = [[1, 2, 3], [5, 2, 1, 4], [2, 3, 4, 5, 6]]
final_array = []

for i in range(len(args)):
    ee = args[i]  # Берем подсписок по индексу без его удаления
    for el in ee:
        # Проверяем наличие el в других подсписках, кроме текущего
        if all(el not in args[j] for j in range(len(args)) if j != i):
            if el not in final_array:
                final_array.append(el)

print(final_array)


# In[8]:


def my_function(n):
    if n == 1:
        return '1'
    else:
        return my_function(n - 1) + ' ' + str(n)

print(my_function(5))

#function my_function(n) {
  #  if (n <= 1) return 1;
 #   return(my_function(n-1)+ " " + n);
#}


# In[ ]:





# In[8]:


# Sorted Two Sum
#Given a sorted array of integers and a target value, determine if there exists two integers in the array that sum up to the target value.

#See if you can solve this in O(N) time and O(1) auxiliary space.


array = [1,2,4,7,8,12,123]
val = 13232


first=0
last=len(array)-1

while array[first]<array[last]:
    #print (array[first], array[last] )
    if val> array[first] + array[last]:# need to growth up the sum
        first+=1
    elif val<array[first] + array[last]:# need to growth down the sum
        last-=1
    elif val == array[first] + array[last]:
        print ('yes', array[first],array[last] )
        break

if val != array[first] + array[last]:
    print ('no' )




# In[ ]:


def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return True
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return False

# Пример использования
arr = [1, 2, 4, 5, 6]
target = 10
print(two_sum_sorted(arr, target))


# In[ ]:


array1= [1,3,5]
array2= [2,4,6,8,10]

def merged_array(array1, array2):
    left1,left2=0,0
    result = []
    result_len=len(array1) + len(array2)-1
    
    for i in range(result_len):
        if array1[left1]<array2[left2]:
            result+=array1[left1]
            left1+=1
        elif array1[left1]>array2[left2]:
            result+=array2[left2]
            left2+=1
        else:
            result+=array2[left2]
            result+=array1[left1]
            i+=1
    
    


# In[14]:


array1= [1,3,5]
array2= [2,4,6,8,10]
minlen=min(len(array1),len(array2))
left1,left2=0,0
result = []
result_len=len(array1) + len(array2)-1
    
for _ in range(result_len):
    if array1[left1]<array2[left2]:
        result.append(array1[left1])
        left1+=1
    elif array1[left1]>array2[left2]:
        result.append(array2[left2])
        left2+=1
    else:
        result.append(array1[left1])
        result.append(array2[left2])
        result_len-=1
        


# In[16]:


array1= [1,3,5,7,8,9,10,11,12,13,14,15,16,18,54]
array2= [2,3,4,5,6,8,10,23,25,106]



left1,left2=0,0

result = []



    
while left1<len(array1) and left2<len(array2):
        
    if array1[left1]==array2[left2]:
        result.append (array1[left1])
        
        result.append (array2[left2])
       
        left1+=1
        left2+=1
        
      
    else:
        mini=min(array1[left1],array2[left2])
        result.append (mini)
        i+=1
        

        if array1[left1]<array2[left2]:
            left1+=1
        else:
            left2+=1     


result+=array1[left1:]
result+=array2[left2:]


            
print (result)            


# In[18]:


#Given two sorted arrays of integers, combine the values into one sorted array
def merged_array(array1, array2):
    left1,left2=0,0
    result = []
    while left1<len(array1) and left2<len(array2):
        
        if array1[left1]==array2[left2]:
            result.append (array1[left1])
        
            result.append (array2[left2])
       
            left1+=1
            left2+=1
        
      
        else:
            mini=min(array1[left1],array2[left2])
            result.append (mini)
            
        
            if array1[left1]<array2[left2]:
                left1+=1
            else:
                left2+=1     


    result+=array1[left1:]
    result+=array2[left2:]


            
    print (result)            

array1= [1,3,5,7,8,9,10,11,12,13,14,15,16,18,54]
array2= [2,3,4,5,6,8,10,23,25,106]
merged_array(array1, array2)


# In[21]:


#Given two sorted arrays of integers, combine the values into one sorted array
# path through while i + j < result_lenght:
def merged_array(array1, array2):
    i,j=0,0
    result = []
    result_lenght=len(array1)+len(array2)
    while i + j < result_lenght:
        
        if j>=len(array2) or (i < len(array1) and array1[i]<array2[j]):
            result.append (array1[i])
            i+=1
          
        else:
            result.append (array2[j])
            j+=1
            
     


    result+=array1[i:]
    result+=array2[j:]


            
    print (result)            

array1= [1,3,5,7,8,9,10,11,12,13,14,15,16,18,54]
array2= [2,3,4,5,6,8,10,23,25,106]
merged_array(array1, array2)


# In[37]:


#Two Sum 
#Given an array of integers, and a target value/ determine if there are two integers that add to the sum.
#through set

arr= [4,2,6,5,7,9,10]
x = 13

arrset=set([4,2,6,5,7,9,10])
for i in arrset:
    y=x-i
    if y in arrset:
        print (y,i)
        break
    
        


# In[48]:


#Two Sum 
#Given an array of integers, and a target value/ determine if there are two integers that add to the sum.
#through dict

arr= [4,2,6,5,7,9,10]
x = 13
dict_arr={}

for i in arr:
    dict_arr[i]=0

for k in dict_arr.keys():
    y=x-k
    if y in dict_arr.keys():
        print (y,k)
        break

    


# In[49]:


#Two Sum 
#Given an array of integers, and a target value/ determine if there are two integers that add to the sum.
#through dict wint GPT corrections

arr = [4, 2, 6, 5, 7, 9, 10]
x = 13
dict_arr = {}

for i in arr:
    dict_arr[i] = 0

for k in dict_arr:
    y = x - k
    if y in dict_arr:
        print(y, k)
        break


# In[7]:


#Sort a Bit Array UNEFFECTIVE 
#Given a bit array, return it sorted in-place (a bit array is simply an array that contains only bits, either 0 or 1).
#See if you can solve this in O(N) time and O(1) auxiliary space.
#Try to solve this using a frequency count rather than using multiple pointers, or using a comparison sort function.
#Input : [0, 1, 1, 0, 1, 1, 1, 0]
#Output : [0, 0, 0, 1, 1, 1, 1, 1] 

bit_array = [0, 1, 1, 0, 1, 1, 1, 0]

bit_array_s = [i for i in bit_array if i == 0 ]+[j for j in bit_array if j == 1]
bit_array_s
#However, it involves traversing the list twice, once for the 0s and once for the 1s, 
#which might not be the most efficient in terms of execution time, even though the complexity appears linear.


# In[11]:


#Sort a Bit Array UNEFFECTIVE 
#Given a bit array, return it sorted in-place (a bit array is simply an array that contains only bits, either 0 or 1).
#See if you can solve this in O(N) time and O(1) auxiliary space.
#Try to solve this using a frequency count rather than using multiple pointers, or using a comparison sort function.
#Input : [0, 1, 1, 0, 1, 1, 1, 0]
#Output : [0, 0, 0, 1, 1, 1, 1, 1] 

bit_array = [0, 1, 1, 0, 1, 1, 1, 0]
bit_array_s=[]
for i in range(bit_array.count(0)):
    bit_array_s.append(0)
for i in range(bit_array.count(1)):
    bit_array_s.append(1)
bit_array_s
#However, it involves calling the .count() method twice, each of which traverses the entire array, 
#resulting in O(2N) time complexity


# In[12]:


#Sort a Bit Array 
#Given a bit array, return it sorted in-place (a bit array is simply an array that contains only bits, either 0 or 1).
#See if you can solve this in O(N) time and O(1) auxiliary space.
#Try to solve this using a frequency count rather than using multiple pointers, or using a comparison sort function.
#Input : [0, 1, 1, 0, 1, 1, 1, 0]
#Output : [0, 0, 0, 1, 1, 1, 1, 1] 

bit_array = [0, 1, 1, 0, 1, 1, 1, 0]
bit_array_s=[]
for i in range(bit_array.count(0)):
    bit_array_s.append(0)
quant_1 = len(bit_array)-len(bit_array_s)
for i in range(quant_1):
    bit_array_s.append(1)
bit_array_s


# In[ ]:


#Sort a Bit Array  gpt improve me


bit_array = [0, 1, 1, 0, 1, 1, 1, 0]

# First, count the number of 0s in the original array.
zero_count = bit_array.count(0)

# Then, overwrite the original array with 0s up to the zero_count.
for i in range(zero_count):
    bit_array[i] = 0

# Fill the rest of the array with 1s.
for i in range(zero_count, len(bit_array)):
    bit_array[i] = 1

bit_array


# In[ ]:


#Sort a Bit Array  gpt

bit_array = [0, 1, 1, 0, 1, 1, 1, 0]

# Count the number of 0s in the array.
zero_count = sum(1 for bit in bit_array if bit == 0)

# Fill the first part of the array with 0s based on the count.
for i in range(zero_count):
    bit_array[i] = 0

# Fill the remaining part of the array with 1s.
for i in range(zero_count, len(bit_array)):
    bit_array[i] = 1

bit_array


# In[ ]:


#Binary Search HUETA HERE

#Given a sorted array of unique integers, and a target value determine the index of a matching value within the array. If there is not match, return -1.

arr =  [1,3,4,5,6,7,8,10,11,13,15,17,20,22]
target = 17

arr_div = (len(arr)-1)//2


while arr_div>1:
    if arr[arr_div]<target:
        del arr[:arr_div]
        arr_div=(len(arr)-1)//2
        
    elif arr[arr_div]>target:
        del arr[arr_div:]
        arr_div=(len(arr)-1)//2
        

    elif arr[arr_div]==target:
        print('index',arr_div )
        break
    else:
        print('index',arr_div)


# In[ ]:


#Binary Search 

#Given a sorted array of unique integers, and a target value determine the index of a matching value within the array. If there is not match, return -1.

arr =  [1,3,4,5,6,7,8,10,11,13,15,17,20,22]
target = 14

start=0
end = (len(arr)-1)



while start<=end:
    mid = (start+end)//2
    if target == arr[mid]:
        print('mid!',mid)
        break
    elif target > arr[mid]:
        start=mid
    else:
        end=mid 
if target != arr[mid]:
    print('-1')   


# In[ ]:


#Binary Search cheating

#Given a sorted array of unique integers, and a target value determine the index of a matching value within the array. If there is not match, return -1.

arr =  [1,3,4,5,6,7,8,10,11,13,15,17,20,22]
arr_d = {arr[i]: i for i in range(len(arr))}
target = 15
print(arr_d[target])


# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the range for the input size
n = np.arange(1, 21)

# Calculating complexities
O_1 = np.ones_like(n)
O_log_n = np.log(n)
O_n = n
O_n_log_n = n * np.log(n)
O_n_squared = n**2
O_2_n = 2**n
O_n_factorial = [np.math.factorial(i) for i in n]

# Create a DataFrame
df = pd.DataFrame({
    'O(1)': O_1,
    'O(log n)': O_log_n,
    'O(n)': O_n,
    'O(n log n)': O_n_log_n,
    'O(n^2)': O_n_squared,
    'O(2^n)': O_2_n,
    'O(n!)': O_n_factorial
}, index=n)

# Plotting
plt.figure(figsize=(14, 8))
for complexity in df.columns:
    plt.plot(df.index, df[complexity], label=complexity)

plt.yscale('log')
plt.xlabel('Input Size (n)')
plt.ylabel('Operations (log scale)')
plt.title('Complexity Curves')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




