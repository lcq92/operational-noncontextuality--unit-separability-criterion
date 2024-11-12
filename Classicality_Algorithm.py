import numpy as np
from scipy.linalg import qr
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq as find_root_brentq
from scipy.spatial import ConvexHull
from scipy.integrate import quad
import itertools
from fractions import Fraction
from math import prod
from ppl import Variable, Generator_System, point, ray, C_Polyhedron
from decimal import Decimal
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
from time import time
import random
import pickle
import os


class Algorithm:
    def __init__(self, EPS):
        self.EPS = EPS
        self.decimals = int(-np.log10(EPS))
        
        self.verbose = 0

        self.Rhos     = None
        self.Epsilons = None


        self.rho_zero   = self.bloch_vector_cart_coords_to_rho(0,0,1)
        self.rho_one    = self.bloch_vector_cart_coords_to_rho(0,0,-1)
        self.rho_plus   = self.bloch_vector_cart_coords_to_rho(1,0,0)
        self.rho_minus  = self.bloch_vector_cart_coords_to_rho(-1,0,0)
        self.rho_iplus  = self.bloch_vector_cart_coords_to_rho(0,1,0)
        self.rho_iminus = self.bloch_vector_cart_coords_to_rho(0,-1,0)

        self.eps_trivial_zero = np.array([[0,0],[0,0]])
        self.eps_trivial_one  = np.array([[1,0],[0,1]])
        
        self.rho_plus_exact   = np.array([[0.5,0.5],[0.5,0.5]])
        self.rho_minus_exact  = np.array([[0.5,-0.5],[-0.5,0.5]])
        
        self.rho_iplus_exact  = np.array([[0.5,-0.5j],[0.5j,0.5]])
        self.rho_iminus_exact = np.array([[0.5,0.5j],[-0.5j,0.5]])

    
    @staticmethod
    def hsprod(A,B):
        """Hilbert-Schmidt inner-product of two (Hermitian) square matrices."""
        return np.trace(A @ B)
    
    @staticmethod    
    def inner_product_induced_norm(inner_product):
        """"Returns the norm function associated to a given inner_product function."""
        return lambda vector: np.sqrt( inner_product(vector, vector) )


    def Gram_Schmidt_v1(self, vectors, inner_product):
        """Performs Gram_Schmidt process on k vectors, with arbitrary inner-product.
        
        Note that:
        - it also accepts a linearly-dependent family as input: during the process,
        any zero vector obtained as a consequence of that will be discarded.
        
        
        - the vectors aren't necessarly lists or column arrays. They are any python
        objects, on which inner_product, exterior scalar products and sums are
        well defined. 
        """
        
        induced_norm = self.inner_product_induced_norm(inner_product)
        
        k = len(vectors)
        
        U = [ vectors[0]/induced_norm( vectors[0] ) ]
        
        for i in range(1,k):
            v = vectors[i]
            new_u = v - sum( inner_product(u, v)*u for u in U)
            
            norm_new_u = induced_norm( new_u )
            if norm_new_u < self.EPS:
                pass # discard new vector if it's practically zero
            else:
                U.append( new_u/norm_new_u ) #normalize before adding it to list U
        
        if self.verbose==2:
            if len(U) < k:
                print(f"{k-len(U)} vectors were discarded during a Gram_Schmidt process.")
        return U

    @staticmethod
    def canonical_basis_of_hermitian_matrices(d):
        first_kind = []
        for i in range(d):
            E = np.zeros((d,d), dtype=np.complex128)
            E[i,i] = 1
            first_kind.append(E)

        second_kind = []
        for i in range(1,d):
            for j in range(i):
                E = np.zeros((d,d), dtype=np.complex128)
                E[i,j] = 1 / np.sqrt(2)
                E[j,i] = 1 / np.sqrt(2)
                second_kind.append(E)

        third_kind = []
        for i in range(1,d):
            for j in range(i):
                E = np.zeros((d,d), dtype=np.complex128)
                E[i,j] = 1j   / np.sqrt(2)
                E[j,i] = -1j / np.sqrt(2)
                third_kind.append(E)

        return first_kind+second_kind+third_kind


    def herm2vec(self, H):
        d = H.shape[0]
        herm_orthonomal_basis = self.canonical_basis_of_hermitian_matrices(d)

        return np.array([ self.hsprod(E, H) for E in herm_orthonomal_basis ])

    def vec2herm(self, V):
        d = int(np.sqrt( len(V) ))
        herm_orthonomal_basis = self.canonical_basis_of_hermitian_matrices(d)

        return sum( V[i]*herm_orthonomal_basis[i] for i in range(d**2) )


    def check_orthonormality_and_hermicity(self, herms, raise_error=True):
        
        k = len(herms)

        #construction of matrix of all scalar products
        out = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                out[i,j] = self.hsprod(herms[i], herms[j]).real
                
        
        check_if_orthonormal = np.allclose( out, np.eye(k), atol=10*self.EPS, rtol=0 )
        if not check_if_orthonormal:
            non_orthonormal_error_message = f"Matrices fed in to `check_orthonormality_and_hermicity` are not orthonormal up to 10*A.EPS={10*self.EPS:.1}."
            
            if   raise_error:     raise Exception( non_orthonormal_error_message )
            elif self.verbose==2:           print( non_orthonormal_error_message )
        
        
        check_if_all_are_hermitian =  all( (np.allclose( h.conjugate().transpose(), h, atol=1e-30, rtol=0 ) for h in herms) )
        if not check_if_all_are_hermitian:
            non_all_hermitian_error_message = f"Matrices fed in to `check_orthonormality_and_hermicity` are not all hermitian up to 1e-30."
            
            if   raise_error:     raise Exception( non_all_hermitian_error_message )
            elif self.verbose==2:           print( non_all_hermitian_error_message )
        
        return out


    def Gram_Schmidt(self, herm_matrices, inner_product='', raise_error_if_non_orth_or_herm=True):

        vectors = [self.herm2vec(H) for H in herm_matrices]


        # Create rectangular 2d matrix that has `vectors` as its columns
        a = np.vstack(vectors).T 

        # Compute column-pivoting-QR-decomposition of `a`. 
        Q, R, P = qr(a, pivoting=True) # such that a @ P_mat.T = Q @ R, where P_mat is the
                                       # permutation matrix encoded by the 1D-array P.
      
        bool_arr = np.abs( R.diagonal() ) > self.EPS
        # `bool_arr` should in theory look like [True,...,True, False,....,False].
        # The assertion test below checks that it is of this form:
        false_indices = np.where(bool_arr==False)[0]
        assert (bool_arr[0]) and len(false_indices)==0 or (  len(false_indices)>0 and ( list(false_indices) == list(range(false_indices[0], len(bool_arr))) )  )


        # The number of those `True` values, `r`, corresponds to the rank of A.
        r = len(bool_arr) -  len(false_indices)

        # It's only the first `r` columns of the matrix `Q` that should be kept,
        # their span being equal to the span of the original family of vectors to orthonormalize.
        Q_correct_span = Q[:,:r]
        
        #the column vectors of the matrix `Q_correct_span` are what was seeked.
        orthonormal_vectors = list(Q_correct_span.T)

        
        
        herm_matrices = [self.vec2herm(v) for v in orthonormal_vectors]
        
        if self.verbose==2:
            if len(herm_matrices) < len(vectors):
                print(f"{len(vectors)-len(herm_matrices)} vectors were discarded during a Gram_Schmidt(QR) process.")
        

        self.check_orthonormality_and_hermicity( herm_matrices, raise_error=raise_error_if_non_orth_or_herm )

        return herm_matrices




        
    def project_points_onto_subspace(self, points, subspace_orthonormal_basis, inner_product,
                                     output_as_column_vector=False, discard_duplicates=True):
        """Project a set onto the span of an orthonormal family (for a given inner product).
        
        Optional flags
        --------------
        output_as_column_vector : (default is False) 
            If True, the ouput projected points are given as numpy column vectors
            of their components in the subspace_orthonormal_basis.
            
        discard_duplicates : (default is True)
            Since projecting distinct points can lead to duplicates,
            this will, if True, remove any duplicates.
        """
        
        induced_norm = self.inner_product_induced_norm(inner_product)
        
        if output_as_column_vector:
            out = [ np.array([ inner_product(u,v) for u in subspace_orthonormal_basis]) for v in points ]
            
            norm = np.linalg.norm
            
            # check that at this stage, there are no non-zero imaginary parts in the coefficients
            # (as expected from the supplied real inner_product)
            assert all( norm( x.imag ) < self.EPS for x in out)
            
            # and hence convert the coefficients type from complex to real:
            out = [x.real for x in out]
        
        else:
            out = [ sum( inner_product(u,v)*u for u in subspace_orthonormal_basis ) for v in points ]
            
            norm = induced_norm
            
            
        
        out2 = [x for x in out if not norm(x)<self.EPS]
        
        if self.verbose==2:
            if len(out2) < len(out):
                print(f"{len(out)-len(out2)} null vectors were discarded during a projection process.")

            
        
        if discard_duplicates:
            out3 = []
            for i,vec_b in enumerate(out2):
                
                if all( norm(vec_b-vec_a)>self.EPS for vec_a in out2[:i] ):
                    out3.append(vec_b)  
                   
            if self.verbose==2:    
                if len(out3) < len(out2):
                    print(f"{len(out2)-len(out3)} duplicate vectors were discarded during a projection process.")

            return out3
            
                
        return out2



    @staticmethod
    def sph2cart(r, theta, phi):
        """Convert sperical coordinates of a point in R^3 into cartesian coordinates."""
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return x,y,z


    def bloch_vector_cart_coords_to_rho(self, x, y, z):
        """Computes the qubit density matrix associated to a 3D Bloch vector given as 3 cartesian coordinates."""
        
        # bloch vector should lie in the closed unit ball
        #assert 0-self.EPS <= x**2 + y**2 + z**2 <= 1+self.EPS

        #sqrt(-1)
        i = 1j
        
        rho = 0.5*np.array([[ 1 + z   , x - i*y ],
                             [ x + i*y , 1 - z   ]])
        
  
        rho_rounded = np.round( rho, decimals=self.decimals )
        
        return rho
        
    
    def bloch_vector_sph_coords_to_rho(self, theta, phi, r=1):
        """Computes the qubit density matrix associated to a 3D Bloch vector given as 3 spherical coordinates."""
        x,y,z = self.sph2cart(r, theta, phi)
        
        return self.bloch_vector_cart_coords_to_rho(x,y,z)

    @staticmethod
    def regular_polygon_2d_angles(n_points):
        """Computes the ordered (counter-clockwise) angle coordinates of the `n_points`-sided regular polygon inscribed in the unit circle."""
        return np.linspace(0, 2*np.pi, num=n_points, endpoint=False)

    def generate_polygonal_rhos_in_xy_plane(self, n_points, r=1, phi_offset=0, euler_angles_offset=[0,0,0]):
        """Computed the qubit density matrices associated to Bloch vectors whose tips form an `n_points`-sided regular polygon in the xy-plane."""
        
        xy_plane_sph_coords  = [ [r, np.pi/2, phi] for phi in self.regular_polygon_2d_angles(n_points) ]

        xy_plane_cart_coords = [ self.sph2cart(*sph_coord) for sph_coord in xy_plane_sph_coords ]

        rotation_matrix_3D   = Rotation.from_euler('xyz', euler_angles_offset).as_matrix()

        xy_plane_rotated_cart_coords = [ rotation_matrix_3D @ vec   for vec in xy_plane_cart_coords ]

        return [ self.bloch_vector_cart_coords_to_rho( *rotated_vec ) 
                 for rotated_vec in xy_plane_rotated_cart_coords ] 



    def generate_regular_polyognal_scenario(self, n, r):

        self.Rhos = self.generate_polygonal_rhos_in_xy_plane(n, r=r)

        if n%2 == 0:   #if even number of sides
            self.Epsilons = self.Rhos.copy()

        elif n%2 == 1: #if odd number of sides: must add I-E effect for every effect E present,
                       #which amounts exactly, in this regular-polygonal case, to considering
                       #the same-radius regular-polygon of twice the number of points: 
            self.Epsilons = self.generate_polygonal_rhos_in_xy_plane(2*n, r=r)

    def one_element_completion_of_effects(self, Effects):
        """Takes a list `Effects`of numpy matrices, and returns the twice-longer list {E, I-E for E in Effects}.
           Before outputing, it also discards eventual duplicate matrices (up to A.EPS)."""

        n = len(Effects)
        d = Effects[0].shape[0]


        #some rotation of indices is done here, just so that, when this function is applied to a reg-poly(n odd) effects,
        #it gives the same (output and) order as a reg-poly(2*(n odd)) effects.
        i_afterpi = n//2 + 1
        indices_rotated = np.roll(range(n), -i_afterpi) #rotate the indices of the n points so that it starts at the i_afterpi one.
        New_effects_to_add = [np.eye(d) - Effects[i] for i in indices_rotated]

        def interleave_two_lists(L1, L2):
            return [x for pair in zip(L1, L2) for x in pair]

        completed_Effects = interleave_two_lists( Effects,  New_effects_to_add)


        #check for duplicates
        completed_Effects_dedupl = []
        for i,vec_b in enumerate(completed_Effects):
            
            if all( not np.allclose( vec_a, vec_b, atol=self.EPS, rtol=0 ) for vec_a in completed_Effects[:i] ):
                completed_Effects_dedupl.append(vec_b)  

        if self.verbose==2:    
            if len(completed_Effects_dedupl) < len(completed_Effects):
                print(f"{len(completed_Effects)-len(completed_Effects_dedupl)} duplicate effect matrices were discarded during a completion process.")

        
        return completed_Effects_dedupl





    def one_element_completion_of_angles(self, angles):
        """Takes a list of angles, and returns the twice-longer list {th, th+pi for th in angles}.
           Before outputing, it also discards eventual duplicate angles (up to A.EPS)."""

        n = len(angles)

        New_angles_to_add = [th+np.pi for th in angles]

        completed_angles = angles + New_angles_to_add

        #bring all angles back to [0, 2*pi]:
        completed_angles = np.mod(completed_angles, 2*np.pi)

        #and then sort them
        completed_angles = np.sort(completed_angles)


        #check for duplicates
        completed_angles_dedupl = []
        for i,vec_b in enumerate(completed_angles):
           
           if all( not np.allclose( vec_a, vec_b, atol=self.EPS, rtol=0 ) for vec_a in completed_angles[:i] ):
               completed_angles_dedupl.append(vec_b)  

        if self.verbose==2:    
           if len(completed_angles_dedupl) < len(completed_angles):
               print(f"{len(completed_angles)-len(completed_angles_dedupl)} duplicate angles were discarded during a completion process.")

        return completed_angles_dedupl

    def one_element_completion_of_points(self, points):
            """Takes a list of points (vector in R^n), and returns the twice-longer list {x, -x for x in points}.
               Before outputing, it also discards eventual duplicate points (up to A.EPS)."""

            n = len(points)

            New_points_to_add = [-np.array(x) for x in points]

            completed_points = points + New_points_to_add


            #check for duplicates
            completed_points_dedupl = []
            for i,vec_b in enumerate(completed_points):
               
               if all( not np.allclose( vec_a, vec_b, atol=self.EPS, rtol=0 ) for vec_a in completed_points[:i] ):
                   completed_points_dedupl.append(vec_b)  

            if self.verbose==2:    
               if len(completed_points_dedupl) < len(completed_points):
                   print(f"{len(completed_points)-len(completed_points_dedupl)} duplicate points were discarded during a completion process.")

            return completed_points_dedupl




    @staticmethod
    def generate_random_sorted_angles(n, angles_method='unifglobal'):
        
        if angles_method=='notrandom':
            # exactly regular angles (no randomness) :
            ths = np.linspace(0, 2*np.pi, num=n, endpoint=False)

        if angles_method=='unifglobal':
            # n independent uniform angles: 
            ths = [random.random()*2*np.pi for _ in range(n)]
        
        if angles_method=='unifbins':
            # cut circle into n equal-size bins, and unif sample 1 angle per bin:
            ths = [random.uniform(k*(2*np.pi/n), (k+1)*(2*np.pi/n)) for k in range(n)]

        sorted_ths = sorted(ths)
        return sorted_ths

    @staticmethod
    def generate_random_r(power=1/2):
        u  = random.random()
        return np.float_power(u, power)

     
    def generate_random_polyognal_scenario(self, method='hull', n_s_bounds=[5,5], n_e_bounds=[5,5], output_2d_points_only=True,
                                                 angles_method='unifglobal'):

        n_s = random.randint(*n_s_bounds)
        n_e = random.randint(*n_e_bounds)
            
        if method=='inscribed':
        
            ths_s = self.generate_random_sorted_angles(n_s, angles_method=angles_method)
            ths_e = self.generate_random_sorted_angles(n_e, angles_method=angles_method)
            ths_e = self.one_element_completion_of_angles(ths_e)
            
            r_s = self.generate_random_r(power=1/3)
            r_e = self.generate_random_r(power=1/3)


            the_2d_poly_s = [polar2cart(r_s, th_s) for th_s in ths_s]
            the_2d_poly_e = [polar2cart(r_e, th_e) for th_e in ths_e]


        elif method=='inscribed_unit_radius':
        
            ths_s = self.generate_random_sorted_angles(n_s, angles_method=angles_method)
            ths_e = self.generate_random_sorted_angles(n_e, angles_method=angles_method)
            ths_e = self.one_element_completion_of_angles(ths_e)
            
            r_s = 1
            r_e = 1


            the_2d_poly_s = [polar2cart(r_s, th_s) for th_s in ths_s]
            the_2d_poly_e = [polar2cart(r_e, th_e) for th_e in ths_e]
            
        elif method=='hull':
            
            rand_points_s = sample_random_n_points_in_Rcircle(n_s, R=1)
            rand_points_e = sample_random_n_points_in_Rcircle(n_e, R=1)
            rand_points_e = self.one_element_completion_of_points(rand_points_e)
            

            the_2d_poly_s = convex_hull_of_2d_points(rand_points_s)
            the_2d_poly_e = convex_hull_of_2d_points(rand_points_e)
            
        elif method=='s_rtheta_e_hull':

            # This method uses the same `method=='hull'` procedure for `e`. For `s`, it procedes as follows:
            # - picks a random number n_s
            # - picks n_s random radiuses and n_s random angles
            # - computes 2d convex hull of associated polygon
            # - checks if it contains the origin. If not, restart all of the above.

            # (restricting `s` sets to those that contain the origin can be desirable to investigate trends on purity.)
            
            while True:
                ths_s = self.generate_random_sorted_angles(n_s, angles_method=angles_method)
                rs_s = [self.generate_random_r(power=1/3) for _ in range(n_s)]

                the_2d_poly_s_geners = [polar2cart(r_s, th_s) for r_s, th_s in zip(rs_s,ths_s)]
                the_2d_poly_s,  the_2d_poly_s_ConvexHull_object= convex_hull_of_2d_points(the_2d_poly_s_geners, return_also_ConvexHull_object=True)
                if point_in_2d_convex_hull([0,0], the_2d_poly_s_ConvexHull_object):
                    break

            

            rand_points_e = sample_random_n_points_in_Rcircle(n_e, R=1)
            rand_points_e = self.one_element_completion_of_points(rand_points_e)
            the_2d_poly_e = convex_hull_of_2d_points(rand_points_e)


        if not output_2d_points_only:
            self.Rhos     = [self.bloch_vector_cart_coords_to_rho(x,y,z=0) for x,y in the_2d_poly_s]
            self.Epsilons = [self.bloch_vector_cart_coords_to_rho(x,y,z=0) for x,y in the_2d_poly_e]
        
        return the_2d_poly_s, the_2d_poly_e
        





    def check_validity_of_inputs(self):

        ## Check Rhos and Epsilons aren't empty

        if self.Rhos is None or self.Epsilons is None:
            #raise Exception('\n'.join([ s for s in [self.Rhos, self.Epsilons] if type(s) is str]))
            raise Exception(
                '\n'.join([
                    f"The generating set of {label[0]} has not been set yet."
                    for label in [['States', self.Rhos], ['Effects', self.Epsilons]] if label[1] is None
                    ]))



        ## Check that all elements of Rhos and Epsilons are numpy matrices of the same square shape

        if not all( mat.shape==self.Rhos[0].shape for mat in self.Rhos+self.Epsilons ) and len(set(self.Rhos[0].shape))==1:
            raise Exception('Elements in A.Rhos and A.Epsilons should be numpy arrays having all the same square shape.')

        self.n = self.Rhos[0].shape


    #######################################################################

    def calculate_reduced_space(self):
        
        self.check_validity_of_inputs()
            

        self.Orthonormal_basis_of_spanS = self.Gram_Schmidt(self.Rhos, inner_product=self.hsprod)
        self.Orthonormal_basis_of_spanE = self.Gram_Schmidt(self.Epsilons, inner_product=self.hsprod)

        self.Generating_family_of_R = self.project_points_onto_subspace(self.Orthonormal_basis_of_spanS, self.Orthonormal_basis_of_spanE,
                                                              inner_product=self.hsprod)
        self.Orthonormal_basis_of_R = self.Gram_Schmidt(self.Generating_family_of_R, inner_product=self.hsprod)

        self.Proj_Rhos_on_R     = self.project_points_onto_subspace(self.Rhos, self.Orthonormal_basis_of_R, inner_product=self.hsprod,
                                                  output_as_column_vector=True)
        self.Proj_Epsilons_on_R = self.project_points_onto_subspace(self.Epsilons, self.Orthonormal_basis_of_R, inner_product=self.hsprod,
                                                  output_as_column_vector=True)




    @staticmethod
    def npfrac(x, decimals):
        return Fraction( np.round(x, decimals=decimals) )
    
    

    @staticmethod
    def myfrac(x, decimals):
        y = np.abs(x)
        n = int(y)
        s = int(np.sign(x))
        d = y - n
        dtrim = np.round(d, decimals=decimals)

        numerator = s*( n*(10**decimals)+ int((10**decimals)*dtrim) )
        denominator = 10**decimals

        return Fraction(numerator, denominator)


    
    def myfracv2(self, x, sd, preserve_more_sd_when_denom_is_one=False, pre_eps_filter=False):
        if pre_eps_filter:
            if np.abs(x) < self.EPS:
                return Fraction(0)


        y = np.abs(x)
        n = int(y)
        s = int(np.sign(x))

        shift = int( np.ceil(np.log10(y)) )

        y_shifted = y / 10**shift #'move comma all the way to the left'

        u = np.round(y_shifted, decimals=sd) # first `sd` digits of y(_shifted)
        if sd >= shift:
            numerator = s*  int( u*(10**sd) )
            denominator = 10**(sd-shift)
        else:
            if preserve_more_sd_when_denom_is_one:
                numerator = s*n
                denominator = 1
            else:
                numerator = s*  int( u*(10**shift) )
                denominator = 1
            
        return Fraction(numerator,denominator)

    @staticmethod
    def lcm(L):
        """Lowest Common Multiple of a list of integers."""

        lcm_python_ints = lambda n, m: np.lcm(n, m, dtype=object)
        # Setting dtype=object will make the computation and output be
        # performed using python ints (when the inputs are python ints),
        # instead of the default preliminary conversion to np.int64 type.
        # Keeping python ints is very important since they can be arbitrarily large,
        # while np.int64 cannot and cause overflow problems.
        #
        # Illustration:
        #
        # >>> np.lcm(219060189739591200, 43)
        # 9027155914907130016
        #
        # Expected output is    219060189739591200*43 = 9419588158802421600
        # But we get int overflow (negative wrong result)
        #
        #
        # >>> np.lcm(219060189739591200, 43, dtype=object)
        # 9419588158802421600
        #
        #This time it's the right answer (and the type is still Python int, not np.int64).


        # promote this function back to member of the np.ufunc class, so that ufunc.reduce can be used
        lcm_python_ints = np.frompyfunc(lcm_python_ints, nin=2, nout=1)

        return lcm_python_ints.reduce(L)

    def rationalize_common_denominator(self, V, use_myfracv2=False, do_pre_common_normalization=True, do_lcm=True):
        """Rationalize coefficients (floats) of a vector V, using one common denominator.
        
        Note: . This doesn't return a python Fraction object.
                Rather, it outputs a pair of objects (L,d), where:
                - L is a list of (integer) numerators
                - d is the associated common denominator
                such that  "L[i]/d" is a quotient of integers representing V[i].
              
              . As a preliminary step, all input floats are rounded
                to the nearest `decimals` decimals.
        """
        
        if do_pre_common_normalization:
            V = np.array(V)

            try:    
                max_exponent = np.max( np.ceil(np.log10(np.abs(V[np.abs(V)>=self.EPS]).astype(np.float64))).astype(int).astype(object) )
            except:
                print(V, repr(V), type(V))
                raise Exception()

            V_divided = (V/(10**max_exponent))

            V = V_divided



        if use_myfracv2: 
            Fracs = [self.myfracv2(v, sd=self.decimals) for v in V] #rounding occurs here
        else:
            Fracs = [self.myfrac(v, decimals=self.decimals) for v in V] #rounding occurs here
        
        Numers = [frac.numerator for frac in Fracs]
        Denoms = [frac.denominator for frac in Fracs]

        if do_lcm:
            denoms_lcm = self.lcm([denom for denom in Denoms])
        else:
            denoms_lcm = prod((denom for denom in Denoms))

        V_numerators = [Numers[i]*(denoms_lcm//Denoms[i]) for i in range(len(V))]

        
        if do_pre_common_normalization:
            denoms_lcm /= (10**max_exponent)

        return V_numerators, denoms_lcm


    @staticmethod
    def vec2ppl(v):
        """Turns a vector/list into the pplpy linear polynomial object"""
        dim = len(v)
        E = [Variable(i) for i in range(dim)] #pplpy basis vectors
        return sum( v[i]*E[i] for i in range(dim) )

    def conical_hull(self, ray_vectors, ray_coefficients_are_already_integers=False):
        """Constructs a convex cone (as a pplpy object), given a set of generating vectors."""
        d = len( ray_vectors[0] )
        assert all(len(v)==d for v in ray_vectors)

        
        if ray_coefficients_are_already_integers:
            ray_numerators = ray_vectors
        else:
            ray_numerators = [self.rationalize_common_denominator(v)[0] for v in ray_vectors]
            # (the actual common denominator is not recorded, since rays aren't concerned with normalization)
        
        
        #store `ray_numerators` in object, for debugging
        self.last_ray_numerators = ray_numerators
        
        
        ppl_rays = [ray( self.vec2ppl(v) ) for v in ray_numerators]

        p_zero = point(0)
        gs_cone = Generator_System(p_zero)
        for ppl_ray in ppl_rays:
            gs_cone.insert(ppl_ray)

        Cone = C_Polyhedron(gs_cone)
        
        if self.verbose==2:
            print(Cone)
        
        n_input_generating_rays = len(ray_vectors)
        n_found_extremal_rays = len(Cone.generators()) - 1 #don't count the zero base point
        
        if self.verbose==2:
            if n_found_extremal_rays != n_input_generating_rays:
                print(f"({n_input_generating_rays - n_found_extremal_rays} of the input rays were found [by pplpy] " \
                       "not to be extremal rays of the cone)")
        
        return Cone



    def generate_first_step_cones(self):
        self.Cone_Rhos = self.conical_hull( self.Proj_Rhos_on_R )
        self.Cone_Epsilons = self.conical_hull( self.Proj_Epsilons_on_R )

    def perform_first_step_cones_VEs(self):
        
        # NOTE: Anytime pplpy performs a .(minimized_)constraints() call, to go from Vrep to Hrep,
        # it seems to output integers which can have up to *double* the amount of digits. 
        self.Cone_Rhos_Hrep = self.Cone_Rhos.minimized_constraints()
        self.Cone_Epsilons_Hrep = self.Cone_Epsilons.minimized_constraints()




    def Hrep_to_Vrep_of_polar(self, polhedr_constraints):
        """Takes a pplpy Constraint_System object and returns list of vectors defining the linear inequalities."""
        
        # it is important here to convert the pplpy coefficients to python ints,
        # and not np.int64 or np.float64, to not get integer overflow.
        
        return [ [int(coeff) for coeff in ineq.coefficients()] for ineq in polhedr_constraints ]
        
     


    @staticmethod
    def tensor_product(u,v):
        """Computes coefficients of tensor product of 2 vectors, using lexographical ordering for the product basis."""
        

        # It is important again here to declare vectors as array of 'objects',
        # so that (when u and v have python ints coefficients), the python ints
        # are preserved by the np.tensordot call, otherwise it automatically
        # converts them to np.int64 type arrays, which can overflow.
        u = np.array(u, dtype=object)
        v = np.array(v, dtype=object)
        return np.tensordot(u,v, axes=0).flatten()


    def set_tensor_product(self, A,B):
        """Computes the set of all tensor products of pairs (a,b) in A x B."""
        out = []
        for u in A:
            for v in B:
                out.append( self.tensor_product(u,v) )
        return out


    @staticmethod
    def canonical_basis(dim):
        """Generates list of canonical basis vectors in dimension `dim`."""
        return list(np.eye(dim))


    @staticmethod
    def decimal_sqrt(n):
        """Takes a python int, converts it to Decimal type, and calculate its .sqrt() method.
        
        Note: The output is kept in the Decimal object form, as, for a very large integer input,
        the output will be a Decimal which represents a large float (in scientific notation),
        which may be too large to be converted into a python or numpy float.
        """
        return Decimal(n).sqrt()


    def eucl_norm(self, L):
        """Calculated euclidian norm of a vector, and outputs it in Decimal type."""
        return self.decimal_sqrt( sum(vi*vi for vi in L) )


    def point_constraint_distance(self, point, constraint, normalize='both', round_output=False):
        """Takes a single pplpy Constraint object, and evaluates a pplpy Point object's associated (signed) distance.
        
        Optional arguments
        --------------
        normalize : (default is 'both') 
            Can be any of 'none' (False), 'point', 'constraint', or 'both' (True).
            This argument will dictate how a possible normalization of that signed distance is performed.
            
        round_output : (default is False)
            if True, the signed distance will be rounded, according to the algorithm `self.digits` parameter.
        """
     
        if normalize==0 or normalize is False: normalize = 'none'
        if normalize==1 or normalize is True:  normalize = 'both'
        assert normalize in ['none', 'point', 'constraint', 'both']
        
        
        
        p_coeffs = np.array([int(coeff) for coeff in point.coefficients()],      dtype=object)
        c_coeffs = np.array([int(coeff) for coeff in constraint.coefficients()], dtype=object)
        
        distance = sum( Decimal(c_coeffs[i])*Decimal(p_coeffs[i]) for i in range(len(p_coeffs)) )
        
        
        normalize_factor = 1
        
        if normalize in ['point', 'both']:
            normalize_factor *= self.eucl_norm(p_coeffs)
        else:
            p_coeffs_denominator = int( point.divisor() ) #a pplpy point can in general be rational  (J_ppl has divisor 1 though)
            normalize_factor *= p_coeffs_denominator
        
        if normalize in ['constraint', 'both']:
            normalize_factor *=  self.eucl_norm(c_coeffs)
        else:
            pass
        
        
        
        try: distance /= normalize_factor
        except OverflowError:
            print(distance,normalize_factor)
            print(type(distance),type(normalize_factor))
        
        if normalize=='both':
            # now, in this case, `distance` (in Decimal type) represents a number between -1 and 1,
            # round normalized distance floats to original given precision.
            if round_output:
                distance = np.round(np.float128(distance), decimals=self.decimals)
        
        return distance
        

    ### 2 methdods that test if a point is in a polyhedron [defined by its VRep]:

    @staticmethod
    def point_belongs_to_poly_pplpy_tester(ppl_point, ppl_Poly):
        """The 'direct' method that pplpy provides to test if a point is in a polyhedron."""
        
        return ppl_Poly.contains(C_Polyhedron(ppl_point)) 



    def point_belongs_to_poly_explicit_VE_tester(self, ppl_point, ppl_Poly, return_distances=True, normalize_distances=True, round_distances=False):
        """Test if a point is in a polyhedron, by calculating constraints using V.E., and then testing them.
        
        Returns a tuple (b, L), where b is the test boolean result,
        and L is the list of calculated constraints (signed distances).
        
        Optional flags
        --------------
        return_distances : (default is True) 
            If True, besides returning the test result (boolean), the list of
            signed distances will also be returned.
            
        normalize_distances : (default is True) 
            If True, and if return_distances is True, then signed distances returned
            will be normalized, i.e., each scalar product will be divided by
            - the norm of the ppl_point
            and more importantly, by:
            - the (typically very large) norm of the constraint vector. 
            signed distances will also be returned.
            
        round_distances : (default is False)
            if True, the signed distances will be rounded, according to the algorithm `self.digits` parameter.
        """
        
        #Vertex Enumeration: (Vrep -> Hrep)
        ppl_Poly_Hrep = ppl_Poly.minimized_constraints()
        
        
        point_constraints_distances = [ self.point_constraint_distance(ppl_point, constr, normalize=normalize_distances, round_output=round_distances)
                                       for constr in ppl_Poly_Hrep ]
        
        point_belongs_test = all( distance >= 0 for distance in point_constraints_distances )
        
        if return_distances:  
            return point_belongs_test, np.array(point_constraints_distances)
        else:
            return point_belongs_test
        
        
    def evaluate_closeness_of_generators(self, Cone):
        """Evaluate normalized scalar products of every pair of distinct generator vectors of a Cone."""
        geners = list(Cone.minimized_generators())[1:]
        n_geners = len(geners)
        d = len(geners[0].coefficients())
        pairwise_scalar_products = []
        for i in range(n_geners):
            for j in range(n_geners):
                if i < j:
                    pairwise_scalar_products.append(np.float64( sum( int(geners[i].coefficients()[k])*int(geners[j].coefficients()[k]) for k in range(d) ) / self.decimal_sqrt( sum( int(geners[i].coefficients()[k])*int(geners[i].coefficients()[k]) for k in range(d) )*sum( int(geners[j].coefficients()[k])*int(geners[j].coefficients()[k]) for k in range(d) ) ) ))
        pairwise_scalar_products = np.array(pairwise_scalar_products)
        return pairwise_scalar_products

    def evaluate_closeness_of_inequalities(self, Cone):
        """Evaluate normalized scalar products of every pair of distinct inequality vectors of a Cone."""
        constrs = list(Cone.minimized_constraints())
        n_constrs = len(constrs)
        d = len(constrs[0].coefficients())
        pairwise_scalar_products = []
        for i in tqdm(range(n_constrs)):
            for j in range(n_constrs):
                if i < j:
                    pairwise_scalar_products.append(np.float64( sum( int(constrs[i].coefficients()[k])*int(constrs[j].coefficients()[k]) for k in range(d) ) / self.decimal_sqrt( sum( int(constrs[i].coefficients()[k])*int(constrs[i].coefficients()[k]) for k in range(d) )*sum( int(constrs[j].coefficients()[k])*int(constrs[j].coefficients()[k]) for k in range(d) ) ) ))
        pairwise_scalar_products = np.array(pairwise_scalar_products)
        return pairwise_scalar_products

    



    def generate_second_step_cone(self, second_step_version=1):

        self.Cone_Rhos_Polar_Vrep     = self.Hrep_to_Vrep_of_polar( self.Cone_Rhos_Hrep )
        self.Cone_Epsilons_Polar_Vrep = self.Hrep_to_Vrep_of_polar( self.Cone_Epsilons_Hrep )


        if second_step_version==1:

            self.Sep_extremal_vectors = self.set_tensor_product(self.Cone_Rhos_Polar_Vrep, self.Cone_Epsilons_Polar_Vrep)
            
            self.Cone_Sep = self.conical_hull( self.Sep_extremal_vectors, ray_coefficients_are_already_integers=True )


        elif second_step_version==2:
            
            self.Sep_extremal_vectors = self.set_tensor_product(self.Cone_Rhos_Polar_Vrep, self.Cone_Epsilons_Polar_Vrep)
            self.Sep_extremal_vectors = [V.astype(np.float64) for V in self.Sep_extremal_vectors]

            self.Cone_Sep = self.conical_hull( self.Sep_extremal_vectors )

        elif second_step_version==3:

            self.Cone_Rhos_Polar_Vrep = np.array(self.Cone_Rhos_Polar_Vrep, dtype=np.float64)
            self.Cone_Epsilons_Polar_Vrep = np.array(self.Cone_Epsilons_Polar_Vrep, dtype=np.float64)

            self.Sep_extremal_vectors = self.set_tensor_product(self.Cone_Rhos_Polar_Vrep, self.Cone_Epsilons_Polar_Vrep)
        
            self.Cone_Sep = self.conical_hull( self.Sep_extremal_vectors )

        


        self.J = sum( self.tensor_product(e,e) for e in self.canonical_basis(dim=len(self.Orthonormal_basis_of_R)) )
        self.J_ppl = point(self.vec2ppl(self.J))




    def perform_second_step_cone_calculations(self, fast_check_only=False, normalize_distances=True, round_distances=False):

        pplpy_test_bool_result = self.point_belongs_to_poly_pplpy_tester(self.J_ppl, self.Cone_Sep)
        self.is_classical = pplpy_test_bool_result

        if fast_check_only:
            return None

        pplypy_explicit_test_results = self.point_belongs_to_poly_explicit_VE_tester(self.J_ppl, self.Cone_Sep, normalize_distances=normalize_distances, round_distances=round_distances)
        
        
        
        pplypy_explicit_test_bool_result = pplypy_explicit_test_results[0]

        #raise error if direct and explicit pplpy contains test disagree:
        assert pplpy_test_bool_result is pplypy_explicit_test_bool_result
        
        
        self.J_distances = pplypy_explicit_test_results[1]
        self.n_inequalities = len( self.J_distances )
        self.J_min_distance = np.min( self.J_distances )
        self.signed_distance_to_classicality = -self.J_min_distance
        self.distance_to_classicality = -self.J_min_distance if self.J_min_distance < 0 else 0
        
        
        if self.verbose==2:
            print(f'Number of inequalities: {self.n_inequalities}.')

        if self.verbose>=1:
            print(f"Result of classicality test: {self.is_classical}.")

            print(f"Distance to classicality: {self.distance_to_classicality}. (Minimal constraint value: {self.J_min_distance}.)")


    def max_number_of_digits_seen_in_integers(self):

        print(f"(0) -- input `decimals` parameter: {self.decimals}.") 
        print(f"(a) -- max number of digits in A.Cone_Rhos.minimized_generators(): {np.max( np.vectorize(len)( np.array([np.array([str(abs(x)) for x in el.coefficients()]) for el in self.Cone_Rhos.minimized_generators()]) ) )}.")
        print('')
        print(f"(b) -- max number of digits in A.Cone_Rhos.minimized_constraints(): {np.max( np.vectorize(len)( np.array([np.array([str(abs(x)) for x in el.coefficients()]) for el in self.Cone_Rhos.minimized_constraints()]) ) )}.")
        print(f"(c) -- max number of digits in A.Sep_extremal_vectors: { np.max(np.vectorize(len)(np.vectorize(str)(np.abs(np.array(self.Sep_extremal_vectors))))) }.")
        print(f"(d) -- max number of digits in A.Cone_Sep.minimized_generators(): {np.max( np.vectorize(len)( np.array([np.array([str(abs(x)) for x in el.coefficients()]) for el in self.Cone_Sep.minimized_generators()]) ) )}.")
        print('')
        print(f"(e) -- max number of digits in A.Cone_Sep.minimized_constraints(): {np.max( np.vectorize(len)( np.array([np.array([str(abs(x)) for x in el.coefficients()]) for el in self.Cone_Sep.minimized_constraints()]) ) )}.")

    def run_algorithm(self, fast_check_only=False, verbose=0, normalize_distances=True, round_distances=False, second_step_version=1, print_max_number_of_digits=False):
        self.verbose = verbose
        
        self.calculate_reduced_space()
        self.generate_first_step_cones()
        self.perform_first_step_cones_VEs()
        self.generate_second_step_cone(second_step_version=second_step_version)
        self.perform_second_step_cone_calculations(fast_check_only=fast_check_only, normalize_distances=normalize_distances, round_distances=round_distances)

        if print_max_number_of_digits:
            self.max_number_of_digits_seen_in_integers()

##############################################################################################################################





def sample_random_polar_coords_in_Rcircle(R=1):
    
    u_r  = random.random()
    u_th = random.random()
    
    def r_invcdf(x, R=R): return  R * np.sqrt(x)
    
    r  = r_invcdf(u_r, R=R)
    th = 2*np.pi * u_th
    
    return r, th
    
    
def polar2cart(r,th):
    return r*np.cos(th), r*np.sin(th)

def cart2polar(x,y):
    return np.sqrt(x**2 + y**2), np.arctan2(y,x)
    
    
def sample_random_cart_coords_in_Rcircle(R=1):
    return polar2cart( *sample_random_polar_coords_in_Rcircle(R=R) )

def sample_random_n_points_in_Rcircle(n, R=1):
    return [sample_random_cart_coords_in_Rcircle(R=1) for _ in range(n)]


def regular_polygon_2d(n,r):
    return [polar2cart(r, th) for th in np.linspace(0, 2*np.pi, num=n, endpoint=False)]


def convex_hull_of_2d_points(poly, return_also_ConvexHull_object=False):
    """Takes a list of points in 2D, and returns reduced list of generators of its convex hull,
    with a correct '(counter-clockwise) sorting'."""
    
    ch = ConvexHull(poly)
    
    if return_also_ConvexHull_object:
        return ch.points[ch.vertices], ch

    return ch.points[ch.vertices]

def point_in_2d_convex_hull(point, ConvexHull_object):
    return all( (np.dot(eq[:-1], point) + eq[-1] <= 0) for eq in ConvexHull_object.equations )





def area_of_polygon(poly):
    """Calculates the area of an arbitrary 2D polygon given the ordered list of pairs of (x,y) coordinates.
    (using shoelace formula).
    Credit: https://stackoverflow.com/a/4682656
    """
    x, y = list(zip(*poly))
    
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0


def area_of_regular_polygon(n,r):
    return (n/2)*np.sin(2*np.pi/n)*(r**2)


def perimeter_of_polygon(poly):
    """Calculates the perimeter of an arbitrary 2D polygon given the ordered list of pairs of (x,y) coordinates."""
    return sum( np.linalg.norm( np.array(poly[i+1]) - np.array(poly[i]) ) for i in range(-1, len(poly)-1) )

def perimeter_of_regular_polygon(n,r):
    return (2*n)*np.sin(np.pi/n)*r






def angle_between_2_2d_vectors(v1, v2):
    
    u1 = np.array(v1)/np.linalg.norm(v1)
    u2 = np.array(v2)/np.linalg.norm(v2)
    
    x = np.dot(u1, u2)
    y = u1[0]*u2[1] - u1[1]*u2[0]
    
    return np.arctan2(y,x)



def length_of_attitude_of_triangle(AB_vector, AC_vector, angular_coordinate_on_BC):

    
    AB_vector = np.array(AB_vector)
    AC_vector = np.array(AC_vector)
    BC_vector = -AB_vector + AC_vector

    AC = np.linalg.norm(AC_vector)
    
    A_angle =  abs( angle_between_2_2d_vectors(AB_vector, AC_vector) )
    B_angle =  abs( angle_between_2_2d_vectors(-AB_vector, BC_vector) )
    C_angle =  abs( angle_between_2_2d_vectors(-AC_vector, -BC_vector) )
    
    th = angular_coordinate_on_BC
    ## NB: angular_coordinate_on_BC is expected to lie in interval [0, A_angle]:
    assert 0 <= th <= A_angle
    
    
    r = (AC*np.sin(C_angle)) / np.sin(th + B_angle)
    return r



def create_poly_r_of_theta_function(poly):
    
    poly = np.array(poly)

    poly_ths = [cart2polar(*p)[1] for p in poly]

    #bring all angles back to [0, 2*pi]:
    poly_ths = np.mod(poly_ths, 2*np.pi)


    sorting_idxs = np.argsort(poly_ths)
    sorted_poly      = poly[sorting_idxs]
    sorted_poly_ths  = poly_ths[sorting_idxs]

    def poly_r_of_theta_function(theta):

        #bring input angle back to [0, 2*pi]:
        th = np.mod(theta, 2*np.pi)


        ### Find local reference vector (and 'next one'), and associated local angular coordinate:
        angular_coordinate_reference_vector_index = None
        angle_is_between_0_and_first_vec = False
        for i, poly_th in enumerate(sorted_poly_ths):
            if th < poly_th:
                angular_coordinate_reference_vector_index = i - 1
                if i==0: angle_is_between_0_and_first_vec = True
                break
        if angular_coordinate_reference_vector_index is None:
            angular_coordinate_reference_vector_index = -1

        AB_vector = sorted_poly[angular_coordinate_reference_vector_index]
        AC_vector = sorted_poly[angular_coordinate_reference_vector_index + 1]


        if angle_is_between_0_and_first_vec:
            local_angular_coordinate = th + (2*np.pi - sorted_poly_ths[angular_coordinate_reference_vector_index])
        else:
            local_angular_coordinate = th - sorted_poly_ths[angular_coordinate_reference_vector_index]
        ###

        return length_of_attitude_of_triangle(AB_vector, AC_vector, local_angular_coordinate)


    return poly_r_of_theta_function


def calculate_purity_measures(poly_containing_origin):

    area = area_of_polygon(poly_containing_origin)
    perim = perimeter_of_polygon(poly_containing_origin)
    
    poly_r_of_theta_function = create_poly_r_of_theta_function(poly_containing_origin)
    
    int_r_pow_n = {}
    for n in [3,4,5,6]:
        int_r_pow_n[n], abserr = quad( lambda th: poly_r_of_theta_function(th)**n, 0, 2*np.pi, epsabs=1e-5, epsrel=1e-5)

 
    purity_measures = {}
    
    purity_measures['average_purity'] = 1/2 + int_r_pow_n[4]/(8*area)
    purity_measures['average_purity_on_boundary'] = 1/2 + int_r_pow_n[3]/(2*perim)
    purity_measures['average_purity_per_pol_on_boundary'] = 1/2 + area/(2*np.pi)
    
    purity_measures['variance_purity'] = 1/4 + int_r_pow_n[4]/(8*area) + int_r_pow_n[6]/(24*area)
    purity_measures['variance_purity_on_boundary'] = int_r_pow_n[5]/(4*perim) - ( int_r_pow_n[3]/(2*perim) )**2
    purity_measures['variance_purity_per_pol_on_boundary'] = int_r_pow_n[4]/(8*np.pi) - ( area/(2*np.pi) )**2
    
    
    purity_measures['stdovermean_purity'] = np.sqrt(purity_measures['variance_purity']) / purity_measures['average_purity']
    purity_measures['stdovermean_purity_on_boundary'] = np.sqrt(purity_measures['variance_purity_on_boundary']) / purity_measures['average_purity_on_boundary']
    purity_measures['stdovermean_purity_per_pol_on_boundary'] = np.sqrt(purity_measures['variance_purity_per_pol_on_boundary']) / purity_measures['average_purity_per_pol_on_boundary']
    
    
    def purity(r): return (1/2)*(1 + r**2)
    purity_measures['average_purity_discrete'] = np.mean( [purity(cart2polar(*p)[0]) for p in poly_containing_origin] )
    purity_measures['variance_purity_discrete'] = np.var( [purity(cart2polar(*p)[0]) for p in poly_containing_origin] )
    purity_measures['stdovermean_purity_discrete'] = np.sqrt(purity_measures['variance_purity_discrete']) / purity_measures['average_purity_discrete']
    
    
    
    
    
    return purity_measures





#############################

def humanize_duration(t):
    """Convert number of seconds to some '{HH}h {mm}m {ss}s' string."""
    
    mins_unbounded, secs = divmod(t, 60)
    hours_unbounded, mins = divmod(mins_unbounded, 60)

    mins = int(mins)
    hours_unbounded = int(hours_unbounded)
    
    
    if hours_unbounded==0 and mins==0:
        if secs >= 0.01:
            secs = f"{secs:.2f}" #eg. "0.12"
        else:
            secs = f"{secs:.2e}" #eg. "2.45e-04"
    else:
        secs = round(secs)
        
    if hours_unbounded==0:
        if mins==0:
            return f"{secs}s"
        return f"{mins}m {secs}s"
    return f"{hours_unbounded}h {mins}m {secs}s"




def viz_2d_poly(poly, additional_circles_radiuses=[], poly_color='orange'):
    
    #%config InlineBackend.figure_format = 'retina'
    
    
    poly_Xs, poly_Ys = list(zip(*poly))
    plt.scatter(poly_Xs, poly_Ys, marker='x')
    plt.fill(poly_Xs, poly_Ys, alpha=0.3, color=poly_color)


    t = np.linspace(0,2*np.pi, endpoint=False, num=400)
    circle_x = np.cos(t)
    circle_y = np.sin(t)
    plt.plot(circle_x, circle_y, '--', color='g')

    plt.plot(0,0,'+', color='k', alpha=.3)
    
    if additional_circles_radiuses:
        if not isinstance(additional_circles_radiuses, list): additional_circles_radiuses = [additional_circles_radiuses]
        for r in additional_circles_radiuses:
            plt.plot(r*circle_x, r*circle_y, '--', lw=1)
    

    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.gca().set_aspect('equal', adjustable='box')


    plt.show()


def viz_2d_poly_pair(poly1, poly2, title=None, sqrt2circle=False,
                     poly1_color='xkcd:reddish orange', poly2_color='xkcd:lightish blue', x_color='xkcd:dark purple', unit_color='k',
                     savefigname=False):
    #%config InlineBackend.figure_format = 'retina'



    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage[sc]{mathpazo}'


    poly1_Xs, poly1_Ys = list(zip(*poly1))
    poly2_Xs, poly2_Ys = list(zip(*poly2))


    fig, (ax1, ax2) = plt.subplots(1, 2)



    #remove outer black box
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    #remove ticks
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    ax1.scatter(poly1_Xs, poly1_Ys, marker='x', color=x_color)
    ax1.fill(poly1_Xs, poly1_Ys, alpha=0.3, color=poly1_color)

    ax2.scatter(poly2_Xs, poly2_Ys, marker='x', color=x_color)
    ax2.fill(poly2_Xs, poly2_Ys, alpha=0.3, color=poly2_color)


    t = np.linspace(0,2*np.pi, endpoint=False, num=400)
    circle_x = np.cos(t)
    circle_y = np.sin(t)

    ax1.plot(circle_x, circle_y, '--', color=unit_color, alpha=0.3, linewidth=1.1)
    ax1.plot(0,0,'+', color='k', alpha=.3)

    ax2.plot(circle_x, circle_y, '--', color=unit_color, alpha=0.3, linewidth=1.1)
    ax2.plot(0,0,'+', color='k', alpha=.3)

    if sqrt2circle:
        circle2_x = (1/np.sqrt(2))*np.cos(t)
        circle2_y = (1/np.sqrt(2))*np.sin(t)

        ax1.plot(circle2_x, circle2_y, '--', color=unit_color, alpha=0.3, linewidth=1.1)
        ax1.plot(0,0,'+', color='k', alpha=.3)

        ax2.plot(circle2_x, circle2_y, '--', color=unit_color, alpha=0.3, linewidth=1.1)
        ax2.plot(0,0,'+', color='k', alpha=.3)   
    
    if title:
        ax1.set_ylabel(title, rotation='horizontal', labelpad=25)
    
    ax1.set_xlim(-1.05,1.05)
    ax1.set_ylim(-1.05,1.05)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('$\mathtt{s}$', fontsize=30)         

    ax2.set_xlim(-1.05,1.05)
    ax2.set_ylim(-1.05,1.05)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('$\mathtt{e}$', fontsize=30) 


    plt.tight_layout(pad=2.5)

    if savefigname:
        plt.savefig(savefigname+'.pdf')
        plt.savefig(savefigname+'--tr'+'.pdf', transparent=True)
    plt.show()