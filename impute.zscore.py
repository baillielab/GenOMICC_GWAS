import pandas as pd
import numpy as np
import numba

from os.path import isfile

def arefiles(fileBaseName, exts):
    return all(map(isfile, [fileBaseName + ext for ext in exts]))

class SizeOf:
    """Human friendly memory size"""
    def __init__(self, bytes):
        self.bytes = bytes
    def __repr__(self):
        units = list(zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'], [0, 0, 1, 2, 2, 2]))
        if self.bytes > 1:
            exp  = min(int(math.log(self.bytes, 1024)), len(units) - 1)
            size = self.bytes / 1024**exp
            unit, num_decimals = units[exp]
            return (f'{{:.{num_decimals}f}} {{}}').format(size,unit)
        elif self.bytes == 0:
            return '0 bytes'
        elif self.bytes == 1:
            return '1 byte'

class GenotypesSource:
    def __init__(self, fileName, verbose=False):
        self.fn = str(fileName)
        self.verbose = verbose
        self._load_meta()
        
    def __repr__(self):
        return f"<Genotypes [Individuals={self.shape[1]}, Variants={self.shape[0]}, file={self.fn}]>"

    @staticmethod
    def _detect_format(fileBaseName):
        if arefiles(fileBaseName,['.bed','.bim','.fam']):
            return 'plink-bin'
        if arefiles(fileBaseName,['ped','.map']):
            return 'plink-txt'
        raise ValueError("Could not detect genotype file format.")

    def _load_meta(self, fmt=None):
        if fmt is None:
            fmt = GenotypesSource._detect_format(self.fn)
        impl = {'plink-bin':self._load_meta_plink_bin,
                'plink-txt':self._load_meta_plink_txt
               }
        impl[fmt]()
 
    def _load_meta_plink_txt(self):
        pass
    
    def _load_meta_plink_bin(self):
        self.individuals = pd.read_csv(self.fn + '.fam', header=None, delim_whitespace=True, names=['FID','IID'],usecols=[0,1])
        self.individuals.set_index(['FID','IID'],drop=False,inplace=True)
        self.variants    = pd.read_csv(self.fn + '.bim', header=None, delim_whitespace=True, dtype={'Chr':str,'ID':str,'Morgan':float,'Bp':int,'AlleleA':str,'AlleleB':str},names=['Chr','ID','Morgan','Bp','AlleleA','AlleleB'])
        
        if len(self.variants.ID.unique()) == self.variants.shape[0]:
            self.variants.set_index('ID',drop=False,inplace=True)
        else:
            print("[Note] Variant IDs are not unique. Creating new variant index.")
    
    def _load_data(self, which=slice(None,None,None), who=slice(None,None,None), fmt=None, verbose=False):
        #map range to range, int to range, slice to range, everything else to an 
        if isinstance(which, range):
            pass
        elif isinstance(which, slice):
            which = range(*which.indices(self.variants.shape[0]))
        elif isinstance(which, int):
            which = range(which,which+1)
        elif isinstance(which, pd.Index):
            which = self.variants.index.get_indexer(which)
            if any(-1 == which):
                raise ValueError("")
        else:
            which = np.asarray(which)
            if which.dtype == np.dtype('bool'):
                which = np.flatnonzero(which)
                    
        if isinstance(who, range):
            pass
        elif isinstance(who, slice):
            who = range(*who.indices(self.individuals.shape[0]))
        elif isinstance(who, int):
            who = range(who,who+1)
        elif isinstance(who, pd.Index):
            who = self.individuals.index.get_indexer(who)
            if any(-1 == who):
                raise ValueError("")
        else:
            who = np.asarray(who)
            if who.dtype == np.dtype('bool'):
                who = np.flatnonzero(who)
        
            
        if (len(which) == self.shape[0] and list(which) == range(self.variants.shape[0])): 
            if (len(who) == self.shape[1] and list(who) == range(self.individuals.shape[0])):
                return Genotypes(self.fn, fmt=fmt, verbose=verbose)

        fmt = Genotypes._detect_format(self.fn) if fmt is None else fmt
        impl = {'plink-bin':self._load_data_plink_bin,
                'plink-txt':self._load_data_plink_txt
               }
        return impl[fmt](which, who, verbose=verbose)

    def _load_data_plink_txt(self, which, who, verbose):
        pass
    
    def _load_data_plink_bin(self, which, who, verbose):
        if verbose:
            print(f"Loading genotype data in plink format from {self.fn}.bed.")
            print(f"... loading data of {len(which)} variants for {len(who)} individuals.")
        num_bytes_header  = 3
        num_bytes_per_var = (self.individuals.shape[0]//4 + int((self.individuals.shape[0] % 4) != 0))
        
        G = Genotypes(shape=(len(which),len(who)),verbose=False)
        G.individuals = self.individuals.iloc[who]
        G.variants    = self.variants.iloc[which]
        
        indices = np.empty((len(who),4),dtype=np.dtype('int64'))
        indices[:,0] = who
        indices[:,1] = indices[:,0]%4
        indices[:,0] = indices[:,0]//4
        
        indices[:,2] = range(indices.shape[0])
        indices[:,3] = indices[:,2]%4
        indices[:,2] = indices[:,2]//4
        
        with open(self.fn + '.bed','rb') as f:
            for m,i in enumerate(which):
                dst = G.data[m,:]
                f.seek(num_bytes_header + i*num_bytes_per_var,0)
                src = np.fromfile(f, dtype=np.dtype('int8'), count=num_bytes_per_var)
                impl_filter(dst,src,indices)
        if verbose:
            print(f"... genotypes occupy {SizeOf(G.data.nbytes)}.")
        return G
    
    def __getitem__(self, I):
        if not isinstance(I, tuple):
            I = (I,slice(None,None,None))
        elif not 2 == len(I):
            raise ValueError(f"")
        return self._load_data(which=I[0], who=I[1],verbose=self.verbose)
    
    @property
    def shape(self):
        return (self.variants.shape[0], self.individuals.shape[0])

@numba.guvectorize([(numba.int8[:], numba.int8[:], numba.int64[:,:])], '(n),(k),(N,l)')
def impl_filter(dst, src, i):
    for k in range(i.shape[0]): 
        v = (src[i[k,0]] >> 2*i[k,1]) & 0x03
        dst[i[k,2]] = dst[i[k,2]] | (v << 2*i[k,3])

class Genotypes:
    def __init__(self, fileName=None, shape=None, fmt=None, verbose=False):
        if shape is not None and fileName is not None:
            raise ValueError("Can not specify both shape and fileName.")

        self.data = np.empty((0,0),np.dtype('int8'))
        self.individuals  = pd.DataFrame()
        self.variants     = pd.DataFrame()

        if None is not fileName:
            self.load(fileName, fmt=fmt, verbose=verbose)
        if None is not shape:
            M,N = shape
            num_bytes = M*(N//4 + int((N % 4) != 0))
            if verbose:
                print("Creating zero genotypes.")
                print(f"... genotypes contain {M} variants for {N} individuals.")
                print(f"... genotypes occupy {SizeOf(num_bytes)}.")
            self.data = np.zeros(num_bytes,np.dtype('int8')).reshape((M,-1))
            self.individuals  = pd.DataFrame(index=range(N),columns=['FID','IID'])
            self.individuals.FID = range(N)
            self.individuals.IID = range(N)
            self.variants    = pd.DataFrame(index=range(M),columns=['ID','Chr','Bp','AlleleA','AlleleB'])
            self.variants.ID = range(M)
    def __repr__(self):
        return f"<Genotypes [Individuals={self.shape[1]}, Variants={self.shape[0]}, size={SizeOf(self.data.nbytes)}]>"

    @staticmethod
    def _detect_format(fileBaseName):
        if arefiles(fileBaseName,['.bed','.bim','.fam']):
            return 'plink-bin'
        if arefiles(fileBaseName,['ped','.map']):
            return 'plink-txt'
        raise ValueError("Could not detect genotype file format.")

    def load(self, fileName, fmt=None, verbose=False):
        if fmt is None:
            fmt = Genotypes._detect_format(fileName)
        impl = {'plink-bin':self._load_plink_bin,
                'plink-txt':self._load_plink_txt
               }
        impl[fmt](fileName,verbose=verbose)

    def _load_plink_txt(self, fileName, verbose=False):
        raise NotImplementedError()

    def _load_plink_bin(self, fileName, verbose=False):
        fileInd  = fileName+'.fam'
        fileSnps = fileName+'.bim'
        fileDat  = fileName+'.bed'
        if verbose:
            print(f"Loading genotypes in plink format from {fileName}[.bed,.bim,.fam].")
        self.individuals = pd.read_csv(fileInd, header=None, delim_whitespace=True, names=['FID','IID'],usecols=[0,1])
        self.individuals.set_index(['FID','IID'],drop=False,inplace=True)
        self.variants    = pd.read_csv(fileSnps, header=None, delim_whitespace=True, names=['Chr','ID','Morgan','Bp','AlleleA','AlleleB'])
        
        if len(self.variants.ID.unique()) == self.variants.shape[0]:
            self.variants.set_index('ID',drop=False,inplace=True)
        else:
            if verbose:
                print("[Note] Variant IDs are not unique. Creating new variant index.")
        
        N = self.individuals.shape[0]
        M = self.variants.shape[0]
        if verbose:
            print(f"... genotypes contain {M} variants for {N} individuals.")
        with open(fileDat,'rb') as f:
            #skip header
            f.seek(3)
            #read data
            num_bytes = M*(N//4 + int((N % 4) != 0))
            if verbose:
                print(f"... genotypes occupy {SizeOf(num_bytes)}.")
            self.data = np.fromfile(f, dtype=np.dtype('int8'), count=num_bytes).reshape((M,-1))

    @property
    def shape(self):
        return (self.variants.shape[0], self.individuals.shape[0])

    def freq(self):
        counts = self.count(across='individuals')
        counts['Alleles'] = 2*(counts.Het + counts.HomA + counts.HomB)
        out = self.variants[['ID','Chr','Bp','AlleleA','AlleleB']].copy().set_index('ID')
        #Minor Allele
        A_is_minor = counts.HomA < counts.HomB
        out.loc[ A_is_minor, 'MinorAllele'] = out.AlleleA[A_is_minor]
        out.loc[~A_is_minor, 'MinorAllele'] = out.AlleleB[~A_is_minor]
        #Frequencies for Reference, Minor and Missing Allele
        out['AlleleAFreq'] = (2*counts.HomA + counts.Het)/counts.Alleles
        out.loc[ A_is_minor,'MAF'] = out.AlleleAFreq[A_is_minor]
        out.loc[~A_is_minor,'MAF'] = 1. - out.AlleleAFreq[~A_is_minor]
        out['Missingness'] = counts.Missing/(self.shape[1]*2)
        return out

    def count(self, across):
        if 'individuals' == across:
            return self._count_along()
        elif 'variants' == across:
            return self._count_across()
        else:
            raise ValueError(f"Did not understand value of across = {across}. Expected one of ['variants', 'individuals']")

    def dosage(self, missing=np.nan, ref='A'):
        dose = [0.0,missing,1.0,2.0] if ref == 'A' else [2.0,missing,1.0,0.0]
        code = np.empty((self.shape[0],4))
        code[:,:] = dose
        return EncodedGenotypes(self,code)

    def standardised(self,missing=0.0):
        counts = self.count(across='individuals')
        ref = 2*counts.HomA + counts.Het
        alt = 2*counts.HomB + counts.Het
        
        q = alt/(ref + alt) #frequency of counted allele
        mu  = 2*q
        tau = np.sqrt(2*q*(1.0-q))
        tau[tau == 0.0] = 1.0
        code = np.empty((self.shape[0],4))
        code[:,0] = (0. - mu)/tau
        code[:,2] = (1. - mu)/tau
        code[:,3] = (2. - mu)/tau
        code[:,1] = missing
        
        return EncodedGenotypes(self, code)

    def _count_across(self):
        raise NotImplementedError()

    def _count_along(self):
        K = self.individuals.shape[0] //4
        k = self.individuals.shape[0] % 4
        @numba.guvectorize([(numba.int8[:], numba.int64[:])], '(n),(k)', target='parallel')
        def impl(data,out):
            for i in range(K):
                d = data[i]
                for j in range(4):
                    out[d & 0x03] += 1
                    d >>= 2
            if k > 0:
                d = data[K]
                for j in range(k):
                    out[d & 0x03] += 1
                    d >>= 2
        counts = np.zeros((self.data.shape[0], 4),dtype='int64')
        impl(self.data, counts)
        return pd.DataFrame(counts,index=self.variants.ID, columns=['HomA','Missing','Het','HomB'])

@numba.guvectorize([(numba.int8[:], numba.float64[:], numba.float64[:])], '(n),(k),(N)', target='parallel')
def impl_unpack(data, code, out):
    N = out.shape[0]
    K = N //4
    k = N % 4
    n = 0
    for i in range(K):
        d = data[i]
        for j in range(4):
            out[n] = code[d & 0x03]
            d >>= 2
            n += 1
    if  k > 0:
        d = data[K]
        for j in range(k):
            out[n] = code[d & 0x03]
            d >>= 2
            n += 1

@numba.guvectorize([(numba.int8[:], numba.float64[:], numba.bool_[:], numba.float64[:])], '(n),(k),(M),(N)', target='parallel')
def impl_unpack_mask(data, code, keep, out):
    N = keep.shape[0]
    K = N //4
    k = N % 4
    n = 0
    l = 0
    for i in range(K):
        d = data[i]
        for j in range(4):
            if keep[n]:
                out[l] = code[d & 0x03]
                l += 1
            d >>= 2
            n += 1
    if  k > 0:
        d = data[K]
        for j in range(k):
            if keep[n]:
                out[l] = code[d & 0x03]
                l += 1
            d >>= 2
            n += 1

class EncodedGenotypes:
    def __init__(self, geno, code, which=slice(None,None,None), who=slice(None,None,None)):
        self.geno = geno
        if isinstance(code, pd.DataFrame):
            if self.geno.variants.ID.isin(code.index).all():
                code = np.asarray(code.loc[self.geno.variants.ID, ['HomA','Missing','Het','HomB']].astype(float))
            else:
                missing = ~self.geno.variants.ID.isin(code.index)
                raise ValueError(f"Code is incomplete. Missing information for \
                                   {sum(missing)} variants {list(self.geno.variants.ID[missing])}")
        else:
            code = np.asarray(code)
        if not (code.shape == (self.geno.shape[0],4) or code.shape == (1,4)):
           raise ValueError(f"Code has wrong shape expected {(self.geno.shape[0],4)} or {(1,4)}, got {code.shape}")
        self.code = code

        if isinstance(which, int) and which in range(self.geno.shape[0]):
            self.which = range(which,which+1)
        elif isinstance(which, slice):
            self.which = range(self.geno.shape[0])[which]
        elif isinstance(which, range):
            self.which = range(self.geno.shape[0])[which.start:which.stop:which.step]
        else:
            raise ValueError(f"Only support regular subsets of variants (int, range, slice), got {which}")
        
        if isinstance(who, slice):
            self.who = range(self.geno.shape[1])[who]
        elif isinstance(who, range):
            self.who = range(self.geno.shape[1])[who.start:who.stop:who.step]
        else:
            try:
                who = np.asarray(who)
                if who.dtype is bool:
                    if who.ndim == 1 and len(who) == self.geno.shape[1]:
                        who = np.flatnonzero(who)
                    else:
                        raise ValueError(f"Boolean index for individuals has wrong shape. Expected ({self.geno.shape[1]},), got {who.shape}")
                elif not np.issubdtype(who.dtype, np.integer):
                    raise ValueError(f"Index for individuals has wrong dtype. Expected either bool or integer, got {who.dtype}")
                self.who  = who
            except:
                raise ValueError(f"Did not understand index for individual subset {who}")
        
        self.keep = np.zeros((self.geno.shape[1],),dtype=bool)
        self.keep[who] = True
        if self.keep.all():
            self.keep = None
    @property
    def shape(self):
        return (len(self.which), len(self.who))
    
    @property
    def individuals(self):
        return self.geno.individuals.iloc[self.who]
    
    @property
    def variants(self):
        return self.geno.variants.iloc[self.which]

    def _subindex_individuals(self, I):
        if isinstance(self.who, (slice, range)):
            return np.array(self.who)[I]
        else:
            return self.who[I]

    def __getitem__(self, I):
        if not isinstance(I, tuple):
            I = (I,slice(None,None,None))
        elif not 2 == len(I):
            raise ValueError(f"")
        return EncodedGenotypes(self.geno, self.code, self.which[I[0]], self._subindex_individuals(I[1]))

    def blocks(self, B=100):
        M, N = self.shape
        K = M // B
        b = M %  B
        buffer = np.empty((B,N))
        l,u = 0,B
        for i in range(K):
            self._unpack(slice(l,u),out=buffer)
            yield range(l,u), buffer
            l += B
            u += B
        if b != 0:
            u = M
            self._unpack(slice(l,u),out=buffer[:b,:])
            yield range(l,u), buffer[:b,:]
    
    def asarray(self):
        return self._unpack()

    def asdataframe(self):
        return pd.DataFrame(self.asarray(), index=self.geno.variants.index[self.which], columns=self.geno.individuals.index[self.who])

    def unpack(self):
        return self._unpack()

    def _unpack(self, which=None, out=None):
        if which is None:
            which = slice(0,self.shape[0])
        which = self.which[which]
        #print(which)
        N = self.shape[1]
        K = N //4
        k = N % 4
        M = len(which)

        if out is None:
            out = np.empty((M,N))

        _data = self.geno.data[which, :]
        _code = self.code[which, :] if self.code.shape[0] != 1 else self.code
        if self.keep is not None:
            impl_unpack_mask(_data, _code, self.keep, out)
        else:
            impl_unpack(_data, _code, out)
        return out

def ld(G):
    A = G.standardised().unpack()
    return (A @ A.T)/A.shape[1]

def ld_(G):
    A = G.dosage(missing=0).unpack()
    return np.corrcoef(A)

def impute(target, zscores, geno, delta, ld_min, lam, n_predictors_min = 1, who=slice(None,None,None)):
    chrom = geno.variants.Chr[target]
    bp    = geno.variants.Bp[target]
    
    candidates = (geno.variants.Chr == chrom) & geno.variants.Bp.between(bp-delta,bp+delta) & geno.variants.index.isin(zscores.index)
    
    G = geno[candidates,who]
    S = pd.DataFrame(ld(G),index=G.variants.index,columns=G.variants.index)
    
    predictors = S.index[(np.abs(S.loc[target]) > ld_min) & (S.index != target)]
    if len(predictors) < n_predictors_min:
        return len(predictors), np.nan
    
    S_pp = np.array(S.loc[predictors,predictors])
    S_tp = np.array(S.loc[target,predictors])
    
    w = S_tp @ np.linalg.inv(S_pp + lam*np.eye(S_pp.shape[0])) 
    return len(predictors), (w*np.array(zscores.loc[predictors])).sum()
    
def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fnTargets",type=str,required=True,help='File containing list of target variants. One id per line.')
    parser.add_argument("--fnGeno",type=str,required=True, help='Genotypes in plink format, give stem of [.bed/.bim/.fam] fileset.')
    parser.add_argument("--fnStats",type=str,required=True, help='Tab seperated file with header containing summary stats.')
    parser.add_argument("--col_zscore",default='Tstat',type=str,help='Name of column in stats file with zscore.')
    parser.add_argument("--col_snp",default='SNPID',type=str,help='Name of column in stats file with snp id.')
    parser.add_argument("--delta",default=100000,type=int,help='Maximum distance of predictor variant to target variant.')
    parser.add_argument('--ld_min',default=0.25,type=float,help='Minimum ld between predictor and target variants.')
    parser.add_argument('--lam',default=0.00001,type=float,help='Regularisation parameter.')
    parser.add_argument('--n_predictors_min',default=1,type=int,help='Minumum number of predictors.')
    parser.add_argument('--fnWho',default=None,help='File containing list of individuals in genotype file to use (FID IID pairs one per line, no header). Use all if not given.')
    parser.add_argument('--fnOut',default="imputed.tsv",help='Name of output file containing tab seperated table.')
    
    args = parser.parse_args()
    
    stats  = pd.read_csv(args.fnStats,sep='\t', index_col=args.col_snp, dtype={args.col_snp:str, args.col_zscore:float}, usecols=[args.col_snp, args.col_zscore])[args.col_zscore]
    geno   = GenotypesSource(args.fnGeno)
    
    if args.fnWho is None:
        who = slice(None,None,None)
    else:
        who = pd.read_csv(args.fnWho,delim_whitespace=True,index_col=[0,1],header=None).index
    
    out = pd.DataFrame()
    
    with open(args.fnTargets) as targets:
        for t in targets:
            t = t.strip()
            print(f"Imputing target '{t}'...", end='')
            if t not in geno.variants.index:
                print(f"skipping [not found in genotypes].")
                continue
            n, z = impute(t, stats, geno, args.delta, args.ld_min, args.lam, args.n_predictors_min, who)
            if n < args.n_predictors_min:
                print(f"skipping [Too few predictors (have {n} require {args.n_predictors_min})]")  
                continue
            out = out.append({'target':t, 
                              'n_predictors':n, 
                              'zscore_imputed':z, 
                              'zscore_original':stats[t] if t in stats.index else np.nan}, 
                             ignore_index=True)
            print("done")
    out.set_index('target').to_csv(args.fnOut,sep='\t')
    print(f"Results written to {args.fnOut}")
if __name__ == "__main__":
    main()