
# coding: utf-8

# In[1]:


import multiprocessing as mp


# In[2]:


def job(q):
    res = 0
    for i in range(1000):
        res += i+i**2+i**3
    q.put(res)    # queue


# In[ ]:


if __name__ == '__main__':
    q = mp.Queue()
    p1 = mp.Process(target=job,args=(q,))   # 单值输入加,
    p2 = mp.Process(target=job,args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print(res1+res2)

