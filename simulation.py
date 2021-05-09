import numpy as np
import matplotlib.pyplot as plt

ALPHA =  10**(-9) ## init 10^-10, next 10^-09


def make_init_polynomial(deg):

    ## make polynomial of form from Project handout
    ## PROD_i=1^i=deg (x-i)
    ## useful to have not just as 10 because we can analyze for smaller deg
    ## to find patterns/relationships

    poly = np.poly1d([1])
    for i in range(1, deg+1):
        temp = np.poly1d([1, -i])
        poly = poly * temp
    return poly

def random_perturb_poly(init_poly, alpha):
    coeffs = np.asarray(init_poly)
    new_coeffs = []
    
    for coeff in coeffs:
        new_coeff = coeff * (1 + alpha * np.random.normal()) ## draws from normal dist with mu=0 and sd=1.0=var
        new_coeffs.append(new_coeff)
    
    scared_poly = np.poly1d(new_coeffs)

    return scared_poly

def compute_prob_complex_roots(rounds = 10000, deg=10, alpha = ALPHA):
    print("alpha is", alpha)
    init_poly = make_init_polynomial(deg)
    count = 0

    for i in range(rounds):
        fearful_poly = random_perturb_poly(init_poly, alpha)
        roots = fearful_poly.r 

        real_roots = roots[np.isreal(roots)]
        complex_roots = roots[np.iscomplex(roots)]

        if len(complex_roots) > 0:
            # print("WE HAVE SOME COMPLEX ROOTS")
            # print("All roots", roots)
            # print("Complex roots", complex_roots)
            # print("Real roots", real_roots)
            count += 1
        
    emp_prob = count/rounds
    print("PROB COMPLEX", emp_prob)
    return emp_prob
        



def simul(deg=10, rounds=10000, plot = False, alpha = ALPHA):
    all_roots = [] ## Is nested lists the best way to do this? Dict better?
    for i in range(deg):
        all_roots.append([])

    init_poly = make_init_polynomial(deg)
    init_roots = init_poly.r

    for i in range(rounds):
        fearful_poly = random_perturb_poly(init_poly, alpha)
        roots = fearful_poly.r
        for i, r in enumerate(roots):
            all_roots[i].append(r)
        # all_roots.append(roots)

    print("ROOTS ARE", all_roots)

    if plot == True:
        for i in range(deg):
            og_root = init_roots[i]

            mu = np.mean(all_roots[i])
            sigma = np.std(all_roots[i])

            f = plt.figure(i+1)

            count, bins, ignored = plt.hist(all_roots[i], int(rounds/50), density=True)
            # print("bins are", bins)
            # print("all roots are", all_roots[i])
            plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')

            plt.xlabel(f"Value of Root r{i+1}")
            plt.ylabel("Frequency")
            plt.title(f"Monte Carlo Simulation Values for Root r{i+1} \n Init Root = {og_root:.3f}, mu = {mu:.3f}, sigma = {sigma:.6f}")

        plt.show()

def plot_complexity_rates_by_alpha_deg_2(deg = 2):
    complexity_rates = []
    alphas = []
    for i in range(201):
        val_alpha = i/100.0
        alphas.append(val_alpha)
        temp = compute_prob_complex_roots(deg = deg, alpha = val_alpha)
        complexity_rates.append(temp)
    
    plt.plot(alphas, complexity_rates)
    plt.xlabel("Value of Alpha")
    plt.ylabel("Probability of Petrubing to a Complex Root")
    plt.title("Probability of Introducing a Complex Root for Deg=2 Polynomial based on Value of Alpha")
    plt.show()

def plot_complexity_rates_by_alpha_deg_10(deg = 10):
    complexity_rates = []
    alphas = []
    for i in range(201):
        val_alpha = i/5.0 * 10**(-8)
        alphas.append(val_alpha)
        temp = compute_prob_complex_roots(deg = deg, alpha = val_alpha)
        complexity_rates.append(temp)
    
    plt.plot(alphas, complexity_rates)
    plt.xlabel("Value of Alpha")
    plt.ylabel("Probability of Petrubing to a Complex Root")
    plt.title("Probability of Introducing a Complex Root for \n Deg=10 Polynomial based on Value of Alpha")
    plt.show()



def main():
    a = make_init_polynomial(10)
    print(a)
    # simul(deg=2, plot=True, alpha = ALPHA)
    # compute_prob_complex_roots(deg = 10, alpha = 1*10**(-1))
    # plot_complexity_rates_by_alpha(deg=2)
    # plot_complexity_rates_by_alpha_deg_10()


if __name__ == "__main__":
    main()