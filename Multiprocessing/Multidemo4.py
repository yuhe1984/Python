
# coding: utf-8

# In[1]:


import multiprocessing as mp


# In[2]:


def job(x):
    return x*x


# In[ ]:


def multicore():
    pool = mp.Pool()
    res = pool.map(job,range(10))
    print(res)
    res = pool.apply_async(job,(2,))
    print(res.get())
    multi_res = [pool.apply_async(job,(i,)) for i in range(10)]
    print([res.get() for res in multi_res])


# In[ ]:


if __name__ == '__main__':
    multicore()

