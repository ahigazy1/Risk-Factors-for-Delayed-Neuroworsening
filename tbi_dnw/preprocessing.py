import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ClinicalGCSImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer for GCS variables.
    Synchronized with PUBLICATION_LABELS for robust clinical data handling.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()

        # Define the exact names as they appear AFTER load_and_clean_data renaming
        gcs_total = ['GCS Total (EMS)', 'GCS Total (Admission)', 'GCS Total (Neurosurgery)']
        gcs_motor = ['GCS Motor (EMS)', 'GCS Motor (Admission)', 'GCS Motor (Neurosurgery)']

        # Verify columns exist to avoid KeyErrors
        present_total = [c for c in gcs_total if c in X_out.columns]
        present_motor = [c for c in gcs_motor if c in X_out.columns]

        # 1. TOTAL GCS IMPUTATION
        if 'GCS Total (EMS)' in present_total:
            # Forward fill from clinic/admission if pre-hospital is missing
            if 'GCS Total (Admission)' in present_total:
                X_out['GCS Total (EMS)'] = X_out['GCS Total (EMS)'].fillna(X_out['GCS Total (Admission)'])
            if 'GCS Total (Neurosurgery)' in present_total:
                X_out['GCS Total (EMS)'] = X_out['GCS Total (EMS)'].fillna(X_out['GCS Total (Neurosurgery)'])

        # Bidirectional filling between Admit and NSGY
        if 'GCS Total (Admission)' in present_total and 'GCS Total (Neurosurgery)' in present_total:
            X_out['GCS Total (Admission)'] = X_out['GCS Total (Admission)'].fillna(X_out['GCS Total (Neurosurgery)'])
            X_out['GCS Total (Neurosurgery)'] = X_out['GCS Total (Neurosurgery)'].fillna(X_out['GCS Total (Admission)'])

        # 2. MOTOR GCS IMPUTATION
        if 'GCS Motor (EMS)' in present_motor:
            if 'GCS Motor (Admission)' in present_motor:
                X_out['GCS Motor (EMS)'] = X_out['GCS Motor (EMS)'].fillna(X_out['GCS Motor (Admission)'])
            if 'GCS Motor (Neurosurgery)' in present_motor:
                X_out['GCS Motor (EMS)'] = X_out['GCS Motor (EMS)'].fillna(X_out['GCS Motor (Neurosurgery)'])

        if 'GCS Motor (Admission)' in present_motor and 'GCS Motor (Neurosurgery)' in present_motor:
            X_out['GCS Motor (Admission)'] = X_out['GCS Motor (Admission)'].fillna(X_out['GCS Motor (Neurosurgery)'])
            X_out['GCS Motor (Neurosurgery)'] = X_out['GCS Motor (Neurosurgery)'].fillna(X_out['GCS Motor (Admission)'])

        return X_out
