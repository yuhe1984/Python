
# coding: utf-8

# In[1]:


import multiprocessing as mp
import threading as td


# In[2]:


def job(a,d):
    print('aaaaa')


# In[3]:


t1 = td.Thread(target=job,args=(1,2))
#p1 = mp.Process(target=job,args=(1,2))


# In[7]:


if __name__ == '__main__':
    p1 = mp.Process(target=job,args=(1,2))
    p1.start()
    p1.join()

