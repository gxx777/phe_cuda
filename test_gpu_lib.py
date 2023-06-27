# import cuda_my_add
import sys
import time
import gmpy2

#sys.path.append('./build')
import gpu_lib as gpu_cal

BITS = 2048


def to_32bit_binary_array(value: int):
    binary_str = bin(value)[2:]
    if len(binary_str) < BITS:
        binary_str = '0' * (BITS - len(binary_str)) + binary_str
    return binary_str


def to_32bit_int_array(value: int):
    binary_str = to_32bit_binary_array(value)
    # print(binary_str)
    result = []
    step = 32
    for i in range(0, len(binary_str), step):
        item_binary = binary_str[i:i + step]
        result.append(int(item_binary, 2))
    return result


def get_32bit_int_array(value: int):
    max_len = int(BITS / 32)
    result = [0] * max_len
    for i in range(max_len):
        if i > 0:
            value = value >> 32
        # result[max_len-i-1]=value & 0xffffffff
        result[i] = value & 0xffffffff

        # result.append(value & 0xffffffff)
    result.reverse()
    return result


def binary_array_to_pyint(value):
    pass


# P = 9104787562958695508847332974885348682154901038169717839609090654646128515375101663493652889366834153449213526871419608224084426168937886701076139054339861
# Q = 107551498632982556633886964836542071903484666271679628852389924138175499484129174638370420940298306566038176014252997928628209466950562128041924920386664167302294236358124663341857175964176271284773516601818752621407145273573406652832002925067254463449750498725799441284853986836376973731639405450689418900474
# M = 11567324858200448748355678264458768782932704494788981979368468522159817415459988833506549045313103266625723067614889722058403675565008608565475982405549298911990204557219096845361092388326999746051804552642890509074976380640780008049344823508695157371771506001280500397522856003736103617144446633379961257276034633720324524858527031702577404829262334089420601175470769763115632449536891969500717761707239421793671873052245924816754685324676758280016836732513391943999411924007157162924777215525886724787366976549644463533635466889783331875227194629640520676155079357722966863796165207493787854592185976029400279623729
# # P = 712311
# Q = 4113211231
# M = 712311231123

def generate_big_number():
    pass


def powmod(a, b, c):
    return int(gmpy2.powmod(a, b, c))

def mulmod(a, b, c):
    # import gmpy2
    return int(gmpy2.f_mod(gmpy2.mpz(a) * gmpy2.mpz(b), gmpy2.mpz(c)))


TOTAL = 50000
BATCH_SIZE = 50000


def gpu_cal_time():
    start = time.time()

    batch_param = []

    for i in range(TOTAL):
        p = get_32bit_int_array(P)
        q = get_32bit_int_array(Q)
        m = get_32bit_int_array(M)
        # batch_param.append(gpu_cal.powmod_param_int(p, q, m))
        batch_param.append(gpu_cal.powm_2048(p, q, m))
        if len(batch_param) == BATCH_SIZE:
            print(f'param_time:{time.time() - start}')
    print(f'gpu consume time:{time.time() - start}')


def deal_item(p, q, m):
    a = get_32bit_int_array(p)
    b = get_32bit_int_array(q)
    c = get_32bit_int_array(m)
    return (a, b, c)


def gpu_param_test():
    start = time.time()
    batch_param = []

    for i in range(TOTAL):

        batch_param.append((str(P), str(Q), str(M)))
        if len(batch_param) == BATCH_SIZE:
            print(f'param_time:{time.time() - start}')

            # result = gpu_cal.add_test()
            result = gpu_cal.powm_param_test(batch_param, len(batch_param))
            print(f'gpu_result_len:{result}')
            batch_param = []
    if batch_param:
        result = gpu_cal.powm_2048(batch_param, len(batch_param))
        print(f'gpu_result_len:{len(result)}')
        batch_param = []

    print(f'gpu consume time:{time.time() - start}')


def gpu_param_test2():
    print("enter test2")
    # powm_param_test
    start = time.time()
    batch_param = []

    p = get_32bit_int_array(P)
    q = get_32bit_int_array(Q)
    m = get_32bit_int_array(M)

    for i in range(TOTAL):
        # pool.apply_async()

        # batch_param.append((str(P),str(Q),str(M)))
        # batch_param.append((P,Q,M))
        # batch_param.append((p,q,m))
        batch_param.extend(p)
        batch_param.extend(q)
        batch_param.extend(m)

        if len(batch_param) == BATCH_SIZE * 3 * 64:
            print(f'param_time:{time.time() - start}')

            # result = gpu_cal.add_test()
            result = gpu_cal.powm_param_test2(batch_param, len(batch_param))
            print(f'gpu_result_len:{result}')
            batch_param = []
    # if batch_param:
    #     result = gpu_cal.powm_2048(batch_param, len(batch_param))
    #     print(f'gpu_result_len:{len(result)}')
    #     batch_param = []

    print(f'gpu consume time:{time.time() - start}')


def gpu_param_test4():
    # TOTAL = 1
    batch_param = []
    start = time.time()
    for i in range(TOTAL):
        p_bytes = P.to_bytes(256, "little")
        # print(f"{p_bytes}")
        q_bytes = Q.to_bytes(256, "little")
        # print(f"{q_bytes}")
        m_bytes = M.to_bytes(256, "little")
        # print(f"{m_bytes}")
        batch_param.append((p_bytes, q_bytes, m_bytes))
        if i == 0:
            print("p", int.from_bytes(p_bytes[0:4], "little"))
            print("q", int.from_bytes(q_bytes[0:4], "little"))
            print("m", int.from_bytes(m_bytes[0:4], "little"))
    # print(batch_sparam)
    result = gpu_cal.powm_2048(batch_param, len(batch_param))
    # print("result:",result)
    print(int.from_bytes(result[0], "little"))
    # print(int.from_bytes(result[1], "little"))
    print(f'gpu consume time4:{time.time() - start}')

def gpu_param_test5():
    TOTAL = 1
    batch_param = []
    start = time.time()
    for i in range(TOTAL):
        p_bytes = P.to_bytes(256, "little")
        # print(f"{p_bytes}")
        q_bytes = Q.to_bytes(256, "little")
        # print(f"{q_bytes}")
        m_bytes = M.to_bytes(256, "little")
        # print(f"{m_bytes}")
        batch_param.append((p_bytes, q_bytes, m_bytes))
        if i == 0:
            print("p", int.from_bytes(p_bytes[0:4], "little"))
            print("q", int.from_bytes(q_bytes[0:4], "little"))
            print("m", int.from_bytes(m_bytes[0:4], "little"))
    
    result = gpu_cal.mulm_2048(batch_param, len(batch_param))
    # print("result:",result)
    print(int.from_bytes(result[0], "little"))
    print(f'gpu consume time4:{time.time() - start}')

def cpu_mulm_test():
    start = time.time()
    res = mulmod(P, Q, M)
    print(res)
    print(f'cpu consume time:{time.time() - start}')

def cpu_cal_time():
    start = time.time()
    for i in range(TOTAL):
        # print(i)
        if i == 0:
            res = powmod(P, Q, M)
        else:
            powmod(P, Q, M)
        
    print(res)
    print(f'cpu consume time:{time.time() - start}')


def test_gxx():
    start = time.time()
    # cpu_cal_time()
    # cpu_mulm_test()
    gpu_param_test4()
    # gpu_param_test5()
# test_gxx()