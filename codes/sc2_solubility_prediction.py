from pathlib import Path
from util.utility import MakeFolderWithCurrentFuncName
import pandas as pd
from fastsolv import fastsolv


def sc6RunFastSolv(fd):
    outfd    = MakeFolderWithCurrentFuncName(f'{fd}/results', allow_override=True)
    monomers = pd.read_csv(f'{fd}/data/yasuharasensei_monomers.tsv', sep='\t', index_col=0)
    monomers.rename(columns={'SMILES':'solute_smiles'}, inplace=True)
    monomers['solvent_smiles']='CS(=O)C' # dmso
    monomers['temperature'] = 298 

    predval = fastsolv(monomers)
    print(1)

if __name__ =='__main__':
    bf = Path(__file__).parents[1]
    sc6RunFastSolv(bf)
